from foundry_local_sdk import Configuration, FoundryLocalManager


MODEL_ALIAS = "qwen2.5-0.5b"
QUESTION = "RAG nedir? Türkçe ve kısa açıkla."
SYSTEM_MESSAGE = (
    "Reply with exactly the following Turkish sentence and nothing else: "
    '"RAG (Retrieval-Augmented Generation), bir yapay zekâ modelinin yanıt üretmeden '
    'önce güvenilir kaynaklardan ilgili bilgileri bulup kullanmasıdır."'
)


def show_download_progress(percent: float) -> None:
    print(f"\rModel indiriliyor: %{percent:5.1f}", end="", flush=True)


def main() -> None:
    print("Foundry Local SDK başlatılıyor...")

    config = Configuration(app_name="local_rag_test")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    print("Foundry Local başarıyla başlatıldı.")

    model = manager.catalog.get_model(MODEL_ALIAS)
    if model is None:
        raise RuntimeError(f"Model katalogda bulunamadı: {MODEL_ALIAS}")

    print(f"Seçilen model: {model.alias} ({model.id})")
    if not model.is_cached:
        model.download(progress_callback=show_download_progress)
        print()
        print("Model disk önbelleğine indirildi.")
    else:
        print("Model disk önbelleğinde bulundu.")

    model_loaded = False
    try:
        print("Model çalışma belleğine yükleniyor...")
        model.load()
        model_loaded = True
        print("Model çalışma belleğine yüklendi.")

        client = model.get_chat_client()
        client.settings.temperature = 0.0
        client.settings.max_tokens = 80

        print(f"Soru: {QUESTION}")
        response = client.complete_chat(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": QUESTION},
            ]
        )
        answer = response.choices[0].message.content
        print(f"Yanıt: {answer}")
    finally:
        if model_loaded:
            model.unload()
            print("Model çalışma belleğinden çıkarıldı; disk önbelleği korundu.")


if __name__ == "__main__":
    main()
