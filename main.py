from foundry_local_sdk import Configuration, FoundryLocalManager


def main() -> None:
    print("Foundry Local SDK içe aktarılabildi.")

    config = Configuration(app_name="local_rag_test")
    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance
    print("Foundry Local başarıyla başlatıldı.")
    print("Manager:", manager)


if __name__ == "__main__":
    main()