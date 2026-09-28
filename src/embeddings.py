"""
embeddings.py — Embedding generation and FAISS ANN search.

Uses multilingual-e5-small (118M params, MIT license, 384-dim).
- Requires "passage: " prefix on all input texts (E5 protocol).
- Normalizes embeddings to unit vectors for cosine similarity via dot product.

This module handles:
1. Loading the E5-small model
2. Generating embeddings in batches (memory-safe)
3. Saving/loading embeddings as .npz files
4. Building FAISS IVF index for fast ANN search
5. Querying top-K nearest neighbors

Design: Works on GPU (SageMaker) or CPU (local Mac, just slower).
"""

import os
import gc
import numpy as np
from tqdm import tqdm

from . import config


def load_embedding_model(model_name: str = None, device: str = None):
    """
    Load the sentence-transformers model.
    
    Args:
        model_name: HuggingFace model name. Defaults to config.EMBEDDING_MODEL_NAME.
        device: "cuda", "cpu", or None (auto-detect).
    
    Returns:
        (model, device_str)
    """
    from sentence_transformers import SentenceTransformer
    import torch
    
    if model_name is None:
        model_name = config.EMBEDDING_MODEL_NAME
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading embedding model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    print(f"  Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")
    
    return model, device


def generate_embeddings(
    texts: list,
    entity_ids: list,
    model=None,
    model_name: str = None,
    device: str = None,
    batch_size: int = None,
    prefix: str = None,
    save_path: str = None,
) -> np.ndarray:
    """
    Generate normalized embeddings for a list of texts.
    
    Args:
        texts: List of raw (unnormalized) or normalized business names.
        entity_ids: Corresponding entity IDs (same length as texts).
        model: Pre-loaded SentenceTransformer model (optional).
        model_name: Model name to load if model not provided.
        device: Device to use.
        batch_size: Encoding batch size. None = auto (512 for GPU, 64 for CPU).
        prefix: Text prefix for E5 models. Defaults to config.EMBEDDING_PREFIX.
        save_path: If provided, save embeddings to this .npz file.
    
    Returns:
        (N, D) float32 numpy array of L2-normalized embeddings.
    """
    if prefix is None:
        prefix = config.EMBEDDING_PREFIX
    
    # Prepend E5 prefix
    prefixed_texts = [prefix + str(t) for t in texts]
    
    # Load model if not provided
    own_model = False
    if model is None:
        model, device = load_embedding_model(model_name, device)
        own_model = True
    
    if batch_size is None:
        batch_size = 512 if device == "cuda" else 64
    
    print(f"  Encoding {len(prefixed_texts)} texts (batch_size={batch_size})...")
    embeddings = model.encode(
        prefixed_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize → dot product = cosine sim
        convert_to_numpy=True,
    )
    
    embeddings = embeddings.astype(np.float32)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, ids=np.array(entity_ids), embeddings=embeddings)
        print(f"  Saved embeddings to {save_path} (shape={embeddings.shape})")
    
    if own_model:
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    return embeddings


def load_embeddings(path: str) -> tuple:
    """
    Load embeddings from .npz file.
    
    Returns:
        (ids_array, embeddings_array) where embeddings is (N, D) float32.
    """
    data = np.load(path, allow_pickle=True)
    ids = data["ids"]
    embeddings = data["embeddings"].astype(np.float32)
    print(f"  Loaded embeddings from {path}: {embeddings.shape}")
    return ids, embeddings


def load_embeddings_as_dict(path: str) -> dict:
    """
    Load embeddings as {entity_id: embedding_vector} dict.
    Useful for pairwise cosine lookup during feature computation.
    """
    ids, embeddings = load_embeddings(path)
    return {str(ids[i]): embeddings[i] for i in range(len(ids))}


def build_faiss_index(
    embeddings: np.ndarray,
    entity_ids: np.ndarray,
    nlist: int = None,
    nprobe: int = None,
) -> tuple:
    """
    Build a FAISS IVF index for fast approximate nearest neighbor search.
    
    Uses IVFFlat index: clusters vectors into nlist Voronoi cells,
    then searches nprobe cells at query time.
    
    Args:
        embeddings: (N, D) float32 array, L2-normalized.
        entity_ids: Array of entity IDs (same order as embeddings).
        nlist: Number of Voronoi cells. Default: sqrt(N).
        nprobe: Number of cells to search. Default: config.FAISS_NPROBE.
    
    Returns:
        (index, id_map) where id_map[faiss_idx] = entity_id
    """
    import faiss
    
    n, d = embeddings.shape
    
    if nlist is None:
        nlist = min(int(np.sqrt(n)), 4096)  # Cap at 4096 cells
    if nprobe is None:
        nprobe = config.FAISS_NPROBE
    
    print(f"  Building FAISS IVFFlat index: {n} vectors, dim={d}, nlist={nlist}")
    
    # Use inner product (= cosine sim for normalized vectors)
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Train the index (requires a sample)
    if not index.is_trained:
        train_size = min(n, nlist * 40)  # faiss recommends 30-256 × nlist
        if train_size < n:
            indices = np.random.RandomState(42).choice(n, train_size, replace=False)
            train_data = embeddings[indices]
        else:
            train_data = embeddings
        index.train(train_data)
    
    # Add all vectors
    index.add(embeddings)
    index.nprobe = nprobe
    
    # Build ID mapping (FAISS uses integer indices 0..N-1)
    id_map = {i: str(entity_ids[i]) for i in range(n)}
    
    print(f"  Index built. nprobe={nprobe}, total vectors={index.ntotal}")
    return index, id_map


def faiss_ann_search(
    query_embeddings: np.ndarray,
    query_ids: list,
    index,
    id_map: dict,
    top_k: int = None,
) -> dict:
    """
    Search FAISS index for top-K nearest neighbors per query.
    
    Args:
        query_embeddings: (Q, D) float32 array of S1 embeddings.
        query_ids: List of S1 entity IDs (same order).
        index: FAISS index.
        id_map: {faiss_idx: entity_id} mapping.
        top_k: Number of neighbors to retrieve. Default: config.FAISS_TOP_K.
    
    Returns:
        dict {s1_id: set(candidate_pool_ids)} — candidate pairs from ANN search.
    """
    if top_k is None:
        top_k = config.FAISS_TOP_K
    
    n_queries = len(query_ids)
    print(f"  FAISS ANN search: {n_queries} queries, top_k={top_k}")
    
    candidates = {}
    batch_size = 10000  # Process queries in batches for memory
    
    for start in tqdm(range(0, n_queries, batch_size), desc="FAISS search"):
        end = min(start + batch_size, n_queries)
        batch_q = query_embeddings[start:end]
        batch_ids = query_ids[start:end]
        
        # Search
        scores, indices = index.search(batch_q, top_k)
        
        for i, s1_id in enumerate(batch_ids):
            cands = set()
            for j in range(top_k):
                idx = indices[i, j]
                if idx >= 0:  # -1 means no result
                    pool_id = id_map.get(idx)
                    if pool_id:
                        cands.add(pool_id)
            candidates[s1_id] = cands
    
    total = sum(len(v) for v in candidates.values())
    print(f"  FAISS ANN: {total} total candidates, avg={total/max(n_queries,1):.1f}/entity")
    return candidates


def generate_and_index_country(
    s1_df,
    pool_df,
    country: str,
    model=None,
    model_name: str = None,
    device: str = None,
    save_dir: str = None,
) -> dict:
    """
    End-to-end: generate embeddings + FAISS ANN search for one country.
    
    Args:
        s1_df: S1 DataFrame with 'entity_id' and 'norm_name' columns.
        pool_df: Pool DataFrame with 'entity_id' and 'norm_name' columns.
        country: Country name (for logging/saving).
        model: Pre-loaded model (optional).
        save_dir: Directory to save embeddings (optional).
    
    Returns:
        dict {s1_id: set(candidate_pool_ids)} from FAISS ANN search.
    """
    if save_dir is None:
        save_dir = os.path.join(config.MODEL_DIR, "embeddings")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n--- Embedding + FAISS for {country} ---")
    
    # Load or generate S1 embeddings
    s1_emb_path = os.path.join(save_dir, f"emb_s1_{country}.npz")
    if os.path.exists(s1_emb_path):
        s1_ids, s1_emb = load_embeddings(s1_emb_path)
    else:
        s1_names = s1_df["norm_name"].fillna("").tolist()
        s1_ids_list = s1_df["entity_id"].tolist()
        s1_emb = generate_embeddings(
            s1_names, s1_ids_list, model=model, model_name=model_name,
            device=device, save_path=s1_emb_path,
        )
        s1_ids = np.array(s1_ids_list)
    
    # Load or generate pool embeddings
    pool_emb_path = os.path.join(save_dir, f"emb_pool_{country}.npz")
    if os.path.exists(pool_emb_path):
        pool_ids, pool_emb = load_embeddings(pool_emb_path)
    else:
        pool_names = pool_df["norm_name"].fillna("").tolist()
        pool_ids_list = pool_df["entity_id"].tolist()
        pool_emb = generate_embeddings(
            pool_names, pool_ids_list, model=model, model_name=model_name,
            device=device, save_path=pool_emb_path,
        )
        pool_ids = np.array(pool_ids_list)
    
    # Build FAISS index on pool embeddings
    index, id_map = build_faiss_index(pool_emb, pool_ids)
    
    # Search: find top-K pool neighbors for each S1 entity
    candidates = faiss_ann_search(
        s1_emb, [str(x) for x in s1_ids], index, id_map,
    )
    
    # Cleanup
    del s1_emb, pool_emb, index
    gc.collect()
    
    return candidates
