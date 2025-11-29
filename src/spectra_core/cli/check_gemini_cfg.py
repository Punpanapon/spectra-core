import sys

from spectra_core.util.config import get_env_or_secret, has_env_or_secret


def main():
    key_present = has_env_or_secret("GEMINI_API_KEY")
    model = get_env_or_secret("GEMINI_MODEL", "gemini-2.5-flash")
    print(f"GEMINI_API_KEY present: {key_present}")
    print(f"GEMINI_MODEL: {model}")
    sys.exit(0 if key_present else 0)


if __name__ == "__main__":
    main()
