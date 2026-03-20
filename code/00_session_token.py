import boto3
from dotenv import load_dotenv, set_key
import os
from utils import cred_path

def get_aws_session_token_with_mfa(permanent_env: str = cred_path + "permanent_cred.env",
                                   temporary_env: str = cred_path + "temporary_cred.env",
                                   duration_seconds: int = 3600) -> dict:
    """
    Load permanent keys to generate temporary credentials through STS.
    """

    load_dotenv(permanent_env)

    mfa_serial = os.getenv("AWS_MFA_SERIAL")
    region = os.getenv("AWS_DEFAULT_REGION")

    if not mfa_serial:
        raise ValueError("AWS_MFA_SERIAL non impostato nel .env")

    mfa_code = input("Inserisci codice MFA: ")

    sts = boto3.client("sts", region_name=region)

    resp = sts.get_session_token(
        SerialNumber=mfa_serial,
        TokenCode=mfa_code,
        DurationSeconds=duration_seconds
    )

    creds = resp["Credentials"]

    set_key(temporary_env, "AWS_ACCESS_KEY_ID", creds["AccessKeyId"])
    set_key(temporary_env, "AWS_SECRET_ACCESS_KEY", creds["SecretAccessKey"])
    set_key(temporary_env, "AWS_SESSION_TOKEN", creds["SessionToken"])
    #print(creds["SessionToken"])
    return creds

# If you do not have a valid session token, you can generate one using MFA
creds = get_aws_session_token_with_mfa()