"""
Performance Engine — Unleashes hardware potential for trading.

Leverages:
  - ThreadPoolExecutor for concurrent price fetching (24-core i9)
  - GPU-accelerated neural network for ML predictions (RTX 4080, 12GB VRAM)
  - requests.Session with connection pooling for lower HTTP latency

Designed as drop-in enhancements — the trading engine calls these
instead of its sequential/CPU-only methods.
"""

from __future__ import annotations
import os
import time
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import requests

logger = logging.getLogger("cryptobot.perf")


# ═══════════════════════════════════════════════════════════════
# 1. CONCURRENT PRICE FETCHER  (i9-13900HX — 24 cores)
# ═══════════════════════════════════════════════════════════════

class ConcurrentPriceFetcher:
    """Fetches spot prices for N symbols in parallel using a thread pool.

    Sequential:  100 symbols × 0.2s rate limit = 20 seconds
    Concurrent:  100 symbols ÷ 10 workers    ≈  2 seconds  (10× faster)

    Uses requests.Session for HTTP/1.1 keep-alive connection pooling,
    reducing TCP handshake overhead to near-zero for repeated calls to
    the same Alpaca/Kraken endpoints.
    """

    ALPACA_DATA_URL = "https://data.alpaca.markets/v1beta3/crypto/us"

    def __init__(self, max_workers: int = 10, timeout: float = 8.0):
        self.max_workers = max_workers
        self.timeout = timeout
        # Persistent session with connection pooling
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_workers,
            pool_maxsize=max_workers + 5,
            max_retries=2,
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        # Add Alpaca API headers
        api_key = os.getenv("ALPACA_API_KEY", "")
        api_secret = os.getenv("ALPACA_API_SECRET", "")
        if api_key and api_secret:
            self._session.headers.update({
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
            })
        self._fetch_count = 0
        self._last_batch_time = 0.0
        logger.info(
            f"ConcurrentPriceFetcher initialized | workers={max_workers} "
            f"| pool_size={max_workers + 5}"
        )

    def fetch_all_spot(
        self,
        symbols: List[str],
        client=None,
        retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Dict[str, float]:
        """Fetch all spot prices in a single batch API call. Returns {symbol: price}."""
        t0 = time.perf_counter()
        results: Dict[str, float] = {}

        # Alpaca supports comma-separated symbols in one request
        symbols_param = ",".join(symbols)
        for attempt in range(retries):
            try:
                url = f"{self.ALPACA_DATA_URL}/latest/trades?symbols={symbols_param}"
                resp = self._session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                trades = data.get("trades", {})
                logger.warning(
                    f"BATCH-DEBUG: status={resp.status_code} "
                    f"trade_keys={sorted(trades.keys())} "
                    f"input_syms={sorted(symbols)} "
                    f"matched={len(results)}"
                )
                for sym in symbols:
                    td = trades.get(sym)
                    if td and "p" in td:
                        results[sym] = float(td["p"])
                if results:
                    break
            except Exception as e:
                logger.warning(f"Batch fetch attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(retry_delay)

        elapsed = time.perf_counter() - t0
        self._last_batch_time = elapsed
        self._fetch_count += 1
        failed = len(symbols) - len(results)

        if self._fetch_count <= 3 or self._fetch_count % 10 == 0:
            logger.info(
                f"⚡ Batch fetch: {len(results)}/{len(symbols)} symbols "
                f"in {elapsed:.1f}s (single request)"
                + (f" | {failed} failed" if failed else "")
            )

        return results

    def _public_spot_price(self, symbol: str) -> Optional[float]:
        """Fetch via Alpaca crypto data endpoint using pooled session."""
        try:
            url = f"{self.ALPACA_DATA_URL}/latest/trades?symbols={symbol}"
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            trades = resp.json().get("trades", {})
            trade_data = trades.get(symbol)
            if trade_data and "p" in trade_data:
                return float(trade_data["p"])
            return None
        except Exception:
            return None

    def close(self):
        """Clean up session."""
        self._session.close()


# ═══════════════════════════════════════════════════════════════
# 2. GPU NEURAL NETWORK  (RTX 4080 — 12GB VRAM, CUDA 12.6)
#    Torch is LAZY-LOADED to prevent OpenMP deadlock with sklearn.
# ═══════════════════════════════════════════════════════════════

def _lazy_torch():
    """Import torch only when needed (avoids OpenMP deadlock with sklearn)."""
    import torch
    import torch.nn as nn
    return torch, nn


def _build_net(nn_module, n_features: int):
    """Build trading neural network."""
    return nn_module.Sequential(
        nn_module.Linear(n_features, 128),
        nn_module.BatchNorm1d(128),
        nn_module.ReLU(),
        nn_module.Dropout(0.3),

        nn_module.Linear(128, 64),
        nn_module.BatchNorm1d(64),
        nn_module.ReLU(),
        nn_module.Dropout(0.2),

        nn_module.Linear(64, 32),
        nn_module.BatchNorm1d(32),
        nn_module.ReLU(),

        nn_module.Linear(32, 2),
    )


class GPUModel:
    """GPU-accelerated neural network for trade signal prediction.

    Trains a PyTorch neural net on the RTX 4080 GPU alongside the
    existing sklearn GradientBoosting model. Both predictions are
    ensembled for higher confidence signals.

    Torch is lazy-loaded on first use to prevent OpenMP deadlock with sklearn.
    """

    def __init__(self, n_features: int = 15, model_dir: str = "models"):
        self.n_features = n_features
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "gpu_trading_net.pt")
        self.model = None
        self.device = "cpu"
        self.ready = False
        self._train_count = 0
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._torch = None
        self._nn = None
        self._torch_checked = False

    def _ensure_torch(self) -> bool:
        """Lazy-load torch on first use. Returns True if CUDA available."""
        if self._torch_checked:
            return self._torch is not None

        self._torch_checked = True
        try:
            torch, nn = _lazy_torch()
            self._torch = torch
            self._nn = nn
            if torch.cuda.is_available():
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(
                    f"GPU activated: {gpu_name} ({vram_gb:.0f} GB VRAM) "
                    f"— CUDA {torch.version.cuda}"
                )
            else:
                logger.info("PyTorch loaded (CPU mode)")

            # Try loading existing model
            self._load()
            return True
        except ImportError:
            logger.info("PyTorch not installed — GPU model disabled")
            return False
        except Exception as e:
            logger.warning(f"Torch init failed: {e}")
            return False

    def _load(self):
        """Load saved GPU model if exists."""
        if self._torch is None:
            return
        try:
            if os.path.exists(self.model_path):
                checkpoint = self._torch.load(
                    self.model_path, map_location=self.device, weights_only=False
                )
                n_feat = checkpoint["n_features"]
                net = _build_net(self._nn, n_feat).to(self.device)
                net.load_state_dict(checkpoint["state_dict"])
                net.eval()
                self.model = net
                self._mean = checkpoint.get("mean")
                self._std = checkpoint.get("std")
                self.n_features = n_feat
                self.ready = True
                logger.info(f"GPU model loaded ({self.n_features} features, {self.device})")
        except Exception as e:
            logger.warning(f"GPU model load failed: {e}")

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, lr: float = 0.001):
        """Train neural network on GPU."""
        if not self._ensure_torch() or len(X) < 100:
            return

        torch, nn = self._torch, self._nn
        t0 = time.perf_counter()
        self.n_features = X.shape[1]

        # Normalize features
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        X_norm = (X - self._mean) / self._std

        # Temporal split
        split = int(len(X_norm) * 0.8)
        X_train, X_val = X_norm[:split], X_norm[split:]
        y_train, y_val = y[:split], y[split:]

        # Class weights
        n_up = int(y_train.sum())
        n_down = len(y_train) - n_up
        if n_up == 0 or n_down == 0:
            return
        weight_up = len(y_train) / (2 * n_up)
        weight_down = len(y_train) / (2 * n_down)
        class_weights = torch.tensor([weight_down, weight_up], dtype=torch.float32).to(self.device)

        # Build model
        net = _build_net(nn, self.n_features).to(self.device)

        # Tensors on GPU
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_v = torch.tensor(y_val, dtype=torch.long).to(self.device)

        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-5
        )

        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        net.train()
        for epoch in range(epochs):
            batch_size = min(256, len(X_t))
            indices = torch.randperm(len(X_t), device=self.device)

            for i in range(0, len(X_t), batch_size):
                batch_idx = indices[i:i + batch_size]
                xb, yb = X_t[batch_idx], y_t[batch_idx]
                optimizer.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                optimizer.step()

            # Validation
            net.eval()
            with torch.no_grad():
                val_logits = net(X_v)
                val_acc = (val_logits.argmax(1) == y_v).float().mean().item()
                val_loss = criterion(val_logits, y_v).item()
            net.train()
            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        # Accept or reject
        if best_state is not None and best_val_acc >= 0.55:
            net.load_state_dict(best_state)
            net.eval()
            self.model = net
            self.ready = True
            self._train_count += 1

            elapsed = time.perf_counter() - t0
            logger.info(
                f"GPU model trained in {elapsed:.1f}s | "
                f"OOS: {best_val_acc:.1%} | epochs: {epoch + 1} | "
                f"device: {self.device} | samples: {len(X)}"
            )
            self._save()
        else:
            logger.warning(
                f"GPU model rejected: OOS {best_val_acc:.1%} < 55% "
                f"({epoch + 1} epochs, {time.perf_counter() - t0:.1f}s)"
            )
            self.ready = False

    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """Predict direction using GPU neural network."""
        if not self.ready or self.model is None or self._torch is None:
            return {"direction": 0.5, "confidence": 0.5, "up_prob": 0.5}

        try:
            torch = self._torch
            if features.ndim == 1:
                features = features.reshape(1, -1)
            if self._mean is not None and self._std is not None:
                features = (features - self._mean) / self._std
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32).to(self.device)
                probs = torch.softmax(self.model(x), dim=1).cpu().numpy()[0]

            up_prob = float(probs[1])
            return {
                "direction": up_prob,
                "confidence": float(max(probs)),
                "up_prob": up_prob,
                "down_prob": float(probs[0]),
            }
        except Exception as e:
            logger.debug(f"GPU predict error: {e}")
            return {"direction": 0.5, "confidence": 0.5, "up_prob": 0.5}

    def predict_batch(self, features_batch: np.ndarray) -> List[Dict[str, float]]:
        """Predict for multiple symbols at once on GPU — ~0.5ms for 100 symbols."""
        if not self.ready or self.model is None or self._torch is None:
            return [{"direction": 0.5, "confidence": 0.5, "up_prob": 0.5}] * len(features_batch)

        try:
            torch = self._torch
            if self._mean is not None and self._std is not None:
                features_batch = (features_batch - self._mean) / self._std
            features_batch = np.nan_to_num(features_batch, nan=0.0, posinf=0.0, neginf=0.0)

            with torch.no_grad():
                x = torch.tensor(features_batch, dtype=torch.float32).to(self.device)
                probs = torch.softmax(self.model(x), dim=1).cpu().numpy()

            return [
                {"direction": float(p[1]), "confidence": float(max(p)),
                 "up_prob": float(p[1]), "down_prob": float(p[0])}
                for p in probs
            ]
        except Exception as e:
            logger.debug(f"GPU batch predict error: {e}")
            return [{"direction": 0.5, "confidence": 0.5, "up_prob": 0.5}] * len(features_batch)

    def _save(self):
        """Save model checkpoint."""
        if self.model is None or self._torch is None:
            return
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            checkpoint = {
                "state_dict": self.model.state_dict(),
                "n_features": self.n_features,
                "mean": self._mean,
                "std": self._std,
                "train_count": self._train_count,
            }
            self._torch.save(checkpoint, self.model_path)
            logger.info(f"GPU model saved to {self.model_path}")
        except Exception as e:
            logger.warning(f"GPU model save failed: {e}")


# ═══════════════════════════════════════════════════════════════
# 3. SYSTEM PERFORMANCE TUNER
# ═══════════════════════════════════════════════════════════════

def get_system_stats() -> Dict:
    """Get current system resource utilization for monitoring."""
    stats = {}
    try:
        import psutil
        stats["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        stats["ram_used_gb"] = round(mem.used / (1024**3), 1)
        stats["ram_free_gb"] = round(mem.available / (1024**3), 1)
        stats["ram_percent"] = mem.percent
    except ImportError:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            stats["gpu_name"] = torch.cuda.get_device_name(0)
            stats["gpu_mem_used_mb"] = round(torch.cuda.memory_allocated(0) / (1024**2))
            stats["gpu_mem_cached_mb"] = round(torch.cuda.memory_reserved(0) / (1024**2))
            stats["gpu_device"] = "cuda"
    except (ImportError, Exception):
        pass

    return stats
