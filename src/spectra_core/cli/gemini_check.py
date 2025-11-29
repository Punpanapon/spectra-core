from spectra_core.util.config import get_env_or_secret, has_env_or_secret


def main():
    print("Has GEMINI_API_KEY:", has_env_or_secret("GEMINI_API_KEY"))
    print("Model:", get_env_or_secret("GEMINI_MODEL", "(unset)"))


if __name__ == "__main__":
    main()
