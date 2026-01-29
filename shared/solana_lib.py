from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.keypair import Keypair
from solders.system_program import transfer, TransferParams
from solana.rpc.async_api import AsyncClient
from solders.transaction import Transaction
from .config import settings

client = AsyncClient(settings.SOLANA_RPC_URL)
platform_pubkey = Pubkey.from_string(settings.PLATFORM_WALLET_PUBKEY)

# Load Scheduler Key
try:
    import json
    _key_bytes = json.loads(settings.SCHEDULER_PRIVATE_KEY)
    scheduler_keypair = Keypair.from_bytes(bytes(_key_bytes))
except:
    pass # Handle gracefully if env not set in build context

async def verify_deposit(tx_sig: str) -> int:
    """
    Checks on-chain if PLATFORM_WALLET received SOL in this transaction.
    Returns: Amount in Lamports (0 if invalid).
    """
    try:
        resp = await client.get_transaction(
            Signature.from_string(tx_sig),
            max_supported_transaction_version=0
        )

        if not resp.value: return 0

        meta = resp.value.transaction.meta
        if meta.err: return 0

        # Map accounts to find Platform Wallet index
        account_keys = resp.value.transaction.transaction.message.account_keys
        try:
            # Look through all accounts to find ours
            # Note: In newer transaction versions, logic handles lookup tables,
            # this handles standard transfers.
            idx = -1
            for i, key in enumerate(account_keys):
                if str(key) == str(platform_pubkey):
                    idx = i
                    break

            if idx == -1: return 0

            # Calc diff
            pre = meta.pre_balances[idx]
            post = meta.post_balances[idx]
            return max(0, post - pre)

        except Exception as e:
            print(f"Parsing error: {e}")
            return 0

    except Exception as e:
        print(f"RPC Error: {e}")
        return 0

async def sign_payout(worker_pubkey_str: str, lamports: int) -> str:
    """Sends SOL from Scheduler to Worker"""
    try:
        dest = Pubkey.from_string(worker_pubkey_str)

        # Use transfer function instead of Transfer class
        ix = transfer(TransferParams(
            from_pubkey=scheduler_keypair.pubkey(),
            to_pubkey=dest,
            lamports=lamports
        ))

        # Get latest blockhash
        blockhash_resp = await client.get_latest_blockhash()
        blockhash = blockhash_resp.value.blockhash

        # Create and sign transaction
        tx = Transaction.new_signed_with_payer(
            [ix],
            scheduler_keypair.pubkey(),
            [scheduler_keypair],
            blockhash
        )

        # Send transaction
        resp = await client.send_transaction(tx)
        return str(resp.value)
    except Exception as e:
        print(f"Payout failed: {e}")
        return None
