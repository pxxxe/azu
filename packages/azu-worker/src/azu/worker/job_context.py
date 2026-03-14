import asyncio
from typing import Any, Dict, Tuple, List, Optional, Set


class JobContext:
    """
    Manages the state for a single inference job.
    Tracks topology, peers, layer queues, expert queues, model state, and generation state.
    """

    def __init__(self, job_id: str, topology: List[str] = None):
        self.job_id = job_id

        # Topology for P2P mesh handshake - list of peer P2P URLs
        self.topology: List[str] = topology or []

        # Track which peers have acknowledged handshake
        self.peers_ready: Set[str] = set()

        # Job status: PENDING -> HANDSHAKING -> RUNNING -> COMPLETED
        self.status: str = "PENDING"

        # Event to signal when handshake is complete
        self.handshake_done = asyncio.Event()

        # Layer input queues indexed by layer index
        self.layer_input_queues: Dict[int, asyncio.Queue] = {}

        # Expert input queues indexed by (layer_idx, expert_idx)
        self.expert_input_queues: Dict[Tuple[int, int], asyncio.Queue] = {}

        # Pending expert requests - futures waiting for expert results
        self.pending_expert_requests: Dict[Tuple[int, int], asyncio.Future] = {}

        # Opaque per-job inference state managed entirely by the driver.
        #
        # Created via driver.init_job_state() on the first forward pass and
        # passed back into driver.get_layer_cache(), driver.update_state(), and
        # driver.advance_position() — workers never inspect it directly.
        #
        # Replaces the former single shared kv_cache which was incorrect for
        # hybrid architectures (e.g. Qwen3.5) where different layers require
        # incompatible cache types (DynamicCache vs recurrent state tensors).
        self.model_state: Any = None

        # Generated token IDs for this job
        self.generated_ids: List[int] = []

        # Queue for loopback tokens in autoregressive generation
        self.token_queue: asyncio.Queue = asyncio.Queue()

        # Flag to signal job completion
        self.done: bool = False

        # Auth token issued by the scheduler for this job.
        # Stored here for the P2P server to verify incoming x-auth-token headers.
        self.auth_token: Optional[str] = None

    def get_layer_input_queue(self, layer_idx: int) -> asyncio.Queue:
        """Get or create a queue for receiving layer inputs."""
        if layer_idx not in self.layer_input_queues:
            self.layer_input_queues[layer_idx] = asyncio.Queue()
        return self.layer_input_queues[layer_idx]

    def get_expert_queue(self, layer_idx: int, expert_idx: int) -> asyncio.Queue:
        """Get or create a queue for receiving expert inputs."""
        key = (layer_idx, expert_idx)
        if key not in self.expert_input_queues:
            self.expert_input_queues[key] = asyncio.Queue()
        return self.expert_input_queues[key]

    def mark_peer_ready(self, peer_url: str) -> None:
        """Mark a peer as having completed handshake."""
        self.peers_ready.add(peer_url.rstrip("/"))

    def all_peers_ready(self) -> bool:
        """Check if all peers in topology have completed handshake."""
        if not self.topology:
            return True  # No peers to wait for
        return len(self.peers_ready) >= len(self.topology)

    def reset(self) -> None:
        """Reset the job context for reuse."""
        self.peers_ready.clear()
        self.status = "PENDING"
        self.handshake_done.clear()
        self.layer_input_queues.clear()
        self.expert_input_queues.clear()
        self.pending_expert_requests.clear()
        self.model_state = None
        self.generated_ids.clear()
        self.done = False
        self.auth_token = None
