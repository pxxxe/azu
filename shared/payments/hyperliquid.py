"""
Hyperliquid Payment Provider

Implementation of PaymentProvider for Hyperliquid blockchain.
Hyperliquid is the primary payment layer for azu, enabling fast
and cost-effective transfers for worker payments.
"""

import asyncio
import json
import time
from typing import Optional, Tuple

import aiohttp
from eth_account import Account
from web3 import Web3

from .base import (
    BalanceInfo,
    DepositInfo,
    PaymentProvider,
    PaymentProviderType,
    PayoutInfo,
)


class HyperliquidProvider(PaymentProvider):
    """
    Hyperliquid payment provider implementation.

    Handles deposits, balance queries, and payouts on Hyperliquid.
    Uses the Hyperliquid API for transaction verification and
    the native Web3 for signing transactions.
    """

    def __init__(
        self,
        rpc_url: str,
        address: str,
        private_key: str,
        confirmations_required: int = 1,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the Hyperliquid provider.

        Args:
            rpc_url: Hyperliquid RPC endpoint URL
            address: System wallet address
            private_key: System wallet private key (hex string or JSON array)
            confirmations_required: Number of confirmations needed for deposit
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Delay between retries in seconds
        """
        # FIX: Hyperliquid EVM JSON-RPC lives at /evm, not root
        base = rpc_url.rstrip("/")
        self.rpc_url = base if base.endswith("/evm") else f"{base}/evm"

        self.confirmations_required = confirmations_required
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Parse private key - support both hex string and JSON array formats
        if isinstance(private_key, str):
            if private_key.startswith('['):
                # JSON array format
                key_array = json.loads(private_key)
                self.private_key = bytes(key_array)
            else:
                # Hex format
                self.private_key = private_key
        else:
            self.private_key = private_key

        # Create account from private key
        try:
            self.account = Account.from_key(self.private_key)
        except Exception:
            # If direct key fails, try as bytes
            self.account = Account.from_key(bytes(self.private_key) if isinstance(self.private_key, list) else self.private_key)

        self._address = address.lower() if address else self.account.address.lower()

        # Initialize Web3 connection
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

    @property
    def provider_type(self) -> PaymentProviderType:
        return PaymentProviderType.HYPERLIQUID

    @property
    def chain_symbol(self) -> str:
        return "HYPE"

    def should_payout(self, amount: float) -> bool:
        """
        Hyperliquid has no gas fees — always pay out immediately.
        No threshold batching needed unlike Solana/EVM chains.
        """
        return True

    def get_deposit_address(self) -> str:
        """Returns the Hyperliquid deposit address."""
        return self._address

    async def _make_request(self, method: str, params: list = None) -> dict:
        """
        Make a JSON-RPC request to the Hyperliquid RPC.

        Args:
            method: The RPC method name
            params: The parameters for the method

        Returns:
            The result from the RPC

        Raises:
            Exception: If the request fails
        """
        if params is None:
            params = []

        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "jsonrpc": "2.0",
                        "method": method,
                        "params": params,
                        "id": 1
                    }
                    async with session.post(
                        self.rpc_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"HTTP {response.status}")
                        data = await response.json()
                        if "error" in data:
                            raise Exception(data["error"])
                        return data.get("result")
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise Exception(f"RPC request failed after {self.max_retries} attempts: {e}")

    async def verify_deposit(
        self,
        tx_hash: str,
        expected_sender: Optional[str] = None,
        expected_amount: Optional[float] = None
    ) -> Tuple[bool, Optional[DepositInfo]]:
        """
        Verifies a deposit transaction on Hyperliquid.

        Args:
            tx_hash: The transaction hash to verify
            expected_sender: Optional expected sender address
            expected_amount: Optional expected amount

        Returns:
            Tuple of (is_valid, deposit_info)
        """
        try:
            # Get transaction details from Hyperliquid API
            # First check if this is a Hyperliquid transaction
            # For transfers, we check the transaction directly

            result = await self._make_request("eth_getTransactionByHash", [tx_hash])

            if not result:
                return False, None

            # Check transaction status
            block_number = result.get("blockNumber")
            if not block_number:
                # Transaction not yet mined
                return False, None

            # Convert block number to int
            current_block = await self._make_request("eth_blockNumber")
            current_block_int = int(current_block, 16)
            tx_block_int = int(block_number, 16)
            confirmations = current_block_int - tx_block_int

            if confirmations < self.confirmations_required:
                return False, None

            # Get transaction receipt for more details
            receipt = await self._make_request("eth_getTransactionReceipt", [tx_hash])

            if not receipt or receipt.get("status") != "0x1":
                return False, None

            # Parse transaction data
            tx_from = result.get("from", "").lower()
            tx_to = result.get("to", "").lower()
            tx_value = int(result.get("value", "0x0"), 16)

            # Convert wei to HYPE (assuming 18 decimals)
            tx_amount = tx_value / 1e18

            # Check if transaction is to our address
            if tx_to != self._address:
                return False, None

            # Verify sender if expected
            if expected_sender and tx_from != expected_sender.lower():
                return False, None

            # Verify amount if expected
            if expected_amount and abs(tx_amount - expected_amount) > 0.0001:
                return False, None

            # Get timestamp from block
            block = await self._make_request("eth_getBlockByNumber", [block_number, False])
            timestamp = int(block.get("timestamp", "0x0"), 16)

            deposit_info = DepositInfo(
                tx_hash=tx_hash,
                sender_address=tx_from,
                amount=tx_amount,
                confirmations=confirmations,
                timestamp=timestamp
            )

            return True, deposit_info

        except Exception as e:
            # Log error but don't expose internal details
            print(f"Deposit verification error: {type(e).__name__}")
            return False, None

    async def get_balance(self, address: str) -> BalanceInfo:
        """
        Gets the balance for an address on Hyperliquid.

        Args:
            address: The wallet address to check

        Returns:
            BalanceInfo with available, locked, and total balance
        """
        try:
            address = address.lower()

            # Get native token balance
            result = await self._make_request("eth_getBalance", [address, "latest"])

            if result:
                balance_wei = int(result, 16)
                total = balance_wei / 1e18
            else:
                total = 0.0

            return BalanceInfo(
                available=total,
                locked=0.0,
                total=total
            )

        except Exception:
            return BalanceInfo(available=0.0, locked=0.0, total=0.0)

    async def payout(
        self,
        recipient_address: str,
        amount: float,
        memo: Optional[str] = None
    ) -> PayoutInfo:
        """
        Executes a payout to a recipient via Hyperliquid.

        Args:
            recipient_address: The address to send funds to
            amount: The amount to send in HYPE
            memo: Optional memo (not used for native transfers)

        Returns:
            PayoutInfo with transaction hash and status
        """
        try:
            recipient = recipient_address.lower()

            # Validate recipient address
            if not self.is_address_valid(recipient):
                raise ValueError(f"Invalid recipient address: {recipient_address}")

            # Convert amount to wei
            amount_wei = int(amount * 1e18)

            # Build transaction
            nonce = await self._make_request("eth_getTransactionCount", [self.account.address, "latest"])
            nonce_int = int(nonce, 16)

            chain_id = await self._make_request("eth_chainId")
            chain_id_int = int(chain_id, 16)

            # FIX: Hyperliquid has no gas fees — removed eth_gasPrice fetch
            tx = {
                "nonce": nonce_int,
                "gasPrice": 0,      # No gas fees on Hyperliquid
                "gas": 21000,       # Standard gas for native transfer
                "to": Web3.to_checksum_address(recipient),
                "value": amount_wei,
                "chainId": chain_id_int,
                "data": b"",
            }

            # Sign transaction
            signed_tx = self.account.sign_transaction(tx)

            # FIX: web3.py v6 renamed rawTransaction -> raw_transaction
            # FIX: must include 0x prefix or eth_sendRawTransaction rejects it
            raw_hex = "0x" + signed_tx.raw_transaction.hex()

            # Send transaction
            tx_hash = await self._make_request("eth_sendRawTransaction", [raw_hex])

            # Wait for confirmation
            # Hyperliquid finalises in ~1s so 15s is plenty
            confirmed = False
            for _ in range(15):
                await asyncio.sleep(1)
                receipt = await self._make_request("eth_getTransactionReceipt", [tx_hash])
                if receipt:
                    if receipt.get("status") == "0x1":
                        confirmed = True
                    break

            return PayoutInfo(
                tx_hash=tx_hash,
                recipient_address=recipient,
                amount=amount,
                status="confirmed" if confirmed else "pending"
            )

        except Exception as e:
            raise Exception(f"Payout failed: {str(e)}")

    async def get_transaction_status(self, tx_hash: str) -> Tuple[str, int]:
        """
        Gets the status of a transaction.

        Args:
            tx_hash: The transaction hash to check

        Returns:
            Tuple of (status, confirmations)
        """
        try:
            result = await self._make_request("eth_getTransactionByHash", [tx_hash])

            if not result:
                return "not_found", 0

            block_number = result.get("blockNumber")
            if not block_number:
                return "pending", 0

            current_block = await self._make_request("eth_blockNumber")
            current_block_int = int(current_block, 16)
            tx_block_int = int(block_number, 16)
            confirmations = current_block_int - tx_block_int

            receipt = await self._make_request("eth_getTransactionReceipt", [tx_hash])
            if receipt:
                if receipt.get("status") == "0x1":
                    return "confirmed", confirmations
                else:
                    return "failed", confirmations

            return "pending", confirmations

        except Exception:
            return "error", 0

    def is_address_valid(self, address: str) -> bool:
        """
        Validates an address format for Hyperliquid.

        Args:
            address: The address to validate

        Returns:
            True if address format is valid
        """
        try:
            # Check if it's a valid Ethereum-style address
            if not address:
                return False
            address = address.lower()
            # Check format: 0x followed by 40 hex characters
            if not address.startswith("0x") or len(address) != 42:
                return False
            # Validate hex characters
            int(address[2:], 16)
            return True
        except Exception:
            return False


class HyperliquidTestnetProvider(HyperliquidProvider):
    """
    Hyperliquid testnet provider for development/testing.
    """

    def __init__(
        self,
        rpc_url: str = "https://rpc.hyperliquid-testnet.xyz/evm",
        address: str = None,
        private_key: str = None,
        confirmations_required: int = 1
    ):
        super().__init__(
            rpc_url=rpc_url,
            address=address or "",
            private_key=private_key or "0x" + "00" * 32,
            confirmations_required=confirmations_required
        )
