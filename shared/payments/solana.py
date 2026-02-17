"""
Solana Payment Provider

Implementation of PaymentProvider for Solana blockchain.
Provides compatibility support for Solana-based payments,
allowing the azu network to work with both Hyperliquid and Solana.
"""

import asyncio
import base64
import json
from typing import Optional, Tuple

import aiohttp
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction

from .base import (
    BalanceInfo,
    DepositInfo,
    PaymentProvider,
    PaymentProviderType,
    PayoutInfo,
)


class SolanaProvider(PaymentProvider):
    """
    Solana payment provider implementation.

    Handles deposits, balance queries, and payouts on Solana.
    Uses the Solana JSON-RPC API for transaction verification
    and the solders library for transaction signing.
    """

    def __init__(
        self,
        rpc_url: str,
        address: str,
        private_key: bytes,
        confirmations_required: int = 1,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the Solana provider.

        Args:
            rpc_url: Solana RPC endpoint URL
            address: System wallet address (base58)
            private_key: System wallet private key (bytes)
            confirmations_required: Number of confirmations needed for deposit
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.rpc_url = rpc_url
        self.confirmations_required = confirmations_required
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Parse private key
        if isinstance(private_key, str):
            # Base58 decode
            try:
                self.private_key = base58.b58decode(private_key)
            except Exception:
                # Try as JSON array
                key_array = json.loads(private_key)
                self.private_key = bytes(key_array)
        elif isinstance(private_key, list):
            self.private_key = bytes(private_key)
        else:
            self.private_key = private_key

        # Create keypair
        self.keypair = Keypair.from_bytes(self.private_key)
        self._address = address or str(self.keypair.pubkey())

        # Configure commitment level
        self.commitment = "confirmed"

    @property
    def provider_type(self) -> PaymentProviderType:
        return PaymentProviderType.SOLANA

    @property
    def chain_symbol(self) -> str:
        return "SOL"

    def get_deposit_address(self) -> str:
        """Returns the Solana deposit address."""
        return self._address

    async def _make_request(self, method: str, params: list = None) -> dict:
        """
        Make a JSON-RPC request to the Solana RPC.

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
                        "id": 1,
                        "method": method,
                        "params": params
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
                            raise Exception(str(data["error"]))
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
        Verifies a deposit transaction on Solana.

        Args:
            tx_hash: The transaction signature to verify
            expected_sender: Optional expected sender address
            expected_amount: Optional expected amount in SOL

        Returns:
            Tuple of (is_valid, deposit_info)
        """
        try:
            # Get transaction details
            result = await self._make_request(
                "getTransaction",
                [tx_hash, {"encoding": "jsonParsed", "commitment": self.commitment}]
            )

            if not result:
                return False, None

            # Check confirmation status
            meta = result.get("meta")
            if not meta:
                return False, None

            # Check if transaction was successful
            if meta.get("status", {}).get("Err"):
                return False, None

            # Get confirmations
            confirmations = result.get("confirmationStatus")
            if confirmations != self.commitment and confirmations != "finalized":
                # Check slot-based confirmations
                slot = result.get("slot", 0)
                current_slot = await self._make_request("getSlot", [{"commitment": self.commitment}])
                confirmations = current_slot - slot
                if confirmations < self.confirmations_required:
                    return False, None
            else:
                confirmations = 1  # At least 1 confirmation

            # Parse transaction message
            message = result.get("transaction", {}).get("message", {})
            instructions = message.get("instructions", [])

            # Find transfer instructions
            for instruction in instructions:
                # Check for system program transfer
                if instruction.get("program") == "system":
                    parsed = instruction.get("parsed", {})
                    if parsed.get("type") == "Transfer":
                        info = parsed.get("info", {})
                        destination = info.get("destination")
                        amount_lamports = int(info.get("amount", 0))

                        # Convert lamports to SOL
                        amount_sol = amount_lamports / 1e9

                        # Check destination is our address
                        if destination != self._address:
                            continue

                        # Check sender if expected
                        source = info.get("source")
                        if expected_sender and source != expected_sender:
                            continue

                        # Check amount if expected
                        if expected_amount and abs(amount_sol - expected_amount) > 0.0001:
                            continue

                        # Get timestamp (approximate from block time)
                        block_time = result.get("blockTime", 0)

                        deposit_info = DepositInfo(
                            tx_hash=tx_hash,
                            sender_address=source,
                            amount=amount_sol,
                            confirmations=confirmations,
                            timestamp=block_time
                        )

                        return True, deposit_info

            return False, None

        except Exception as e:
            print(f"Deposit verification error: {type(e).__name__}")
            return False, None

    async def get_balance(self, address: str) -> BalanceInfo:
        """
        Gets the balance for an address on Solana.

        Args:
            address: The wallet address to check (base58)

        Returns:
            BalanceInfo with available, locked, and total balance
        """
        try:
            result = await self._make_request(
                "getBalance",
                [address, {"commitment": self.commitment}]
            )

            if result:
                lamports = result.get("value", 0)
                total = lamports / 1e9
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
        Executes a payout to a recipient via Solana.

        Args:
            recipient_address: The address to send funds to (base58)
            amount: The amount to send in SOL
            memo: Optional memo (not supported in base transaction)

        Returns:
            PayoutInfo with transaction signature and status
        """
        try:
            # Validate recipient address
            if not self.is_address_valid(recipient_address):
                raise ValueError(f"Invalid recipient address: {recipient_address}")

            # Convert amount to lamports
            amount_lamports = int(amount * 1e9)

            # Get recent blockhash
            blockhash_result = await self._make_request(
                "getRecentBlockhash",
                [{"commitment": self.commitment}]
            )
            blockhash = blockhash_result.get("blockhash")

            # Create transfer instruction
            from solders.system_program import Transfer, transfer
            from solders.instruction import Instruction

            transfer_ix = transfer(
                Transfer(
                    from_pubkey=self.keypair.pubkey(),
                    to_pubkey=Pubkey.from_string(recipient_address),
                    lamports=amount_lamports
                )
            )

            # Create transaction
            txn = Transaction()
            txn.add(transfer_ix)
            txn.recent_blockhash = blockhash
            txn.fee_payer = self.keypair.pubkey()

            # Sign transaction
            txn.sign(self.keypair)

            # Serialize transaction
            txn_serialized = txn.serialize()

            # Send transaction
            result = await self._make_request(
                "sendTransaction",
                [
                    base64.b64encode(txn_serialized).decode('utf-8'),
                    {"encoding": "base64", "commitment": self.commitment}
                ]
            )

            tx_hash = result

            # Wait for confirmation
            confirmed = False
            for _ in range(30):  # Wait up to 30 seconds
                await asyncio.sleep(1)
                try:
                    result = await self._make_request(
                        "getTransaction",
                        [tx_hash, {"encoding": "jsonParsed", "commitment": self.commitment}]
                    )
                    if result:
                        if result.get("meta", {}).get("status", {}).get("Ok"):
                            confirmed = True
                        break
                except Exception:
                    continue

            return PayoutInfo(
                tx_hash=tx_hash,
                recipient_address=recipient_address,
                amount=amount,
                status="confirmed" if confirmed else "pending"
            )

        except Exception as e:
            raise Exception(f"Payout failed: {str(e)}")

    async def get_transaction_status(self, tx_hash: str) -> Tuple[str, int]:
        """
        Gets the status of a transaction.

        Args:
            tx_hash: The transaction signature to check

        Returns:
            Tuple of (status, confirmations)
        """
        try:
            result = await self._make_request(
                "getTransaction",
                [tx_hash, {"encoding": "jsonParsed", "commitment": self.commitment}]
            )

            if not result:
                return "not_found", 0

            meta = result.get("meta")
            if not meta:
                return "pending", 0

            if meta.get("status", {}).get("Err"):
                return "failed", 1

            confirmations = result.get("confirmationStatus")
            if confirmations == self.commitment or confirmations == "finalized":
                return "confirmed", 1

            return "pending", 0

        except Exception:
            return "error", 0

    def is_address_valid(self, address: str) -> bool:
        """
        Validates an address format for Solana.

        Args:
            address: The address to validate (base58)

        Returns:
            True if address format is valid
        """
        try:
            if not address:
                return False
            # Try to create a Pubkey - will raise if invalid
            Pubkey.from_string(address)
            return True
        except Exception:
            return False


# Import base58 for key decoding
try:
    import base58
except ImportError:
    # Fallback implementation for base58
    import base64 as _base64
    alphabet = b'123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

    def b58decode(s):
        """Decode a base58 string."""
        if isinstance(s, str):
            s = s.encode('ascii')
        pad = len(s) % 4
        if pad > 0:
            s += b'=' * (4 - pad)
        decoded = _base64.b64decode(s, altchars=b'_-')
        # Custom base58 decoding
        num = 0
        for char in s.rstrip(b'='):
            num = num * 58 + alphabet.index(char)
        result = num.to_bytes((num.bit_length() + 7) // 8, 'big')
        # Add leading zeros
        for char in s:
            if char != alphabet[0]:
                break
            result = b'\x00' + result
        return result

    class base58:
        @staticmethod
        def b58decode(s):
            return b58decode(s)
