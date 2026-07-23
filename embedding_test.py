from foundry_local_sdk import Configuration, FoundryLocalManager


REQUESTED_MODEL_ALIAS = "qwen3-0.6b-embedding"
PREFERRED_MODEL_ALIAS = "qwen3-embedding-0.6b"

SENTENCES = [
    "Staj süresi 20 iş günüdür.",
    "Staj başvuruları mayıs ayında yapılır.",
    "Python bir programlama dilidir.",
    "Öğrenciler staj defterini teslim etmelidir.",
    "Futbol takımı maçı üç golle kazandı.",
]


def show_download_progress(percent: float) -> None:
    print(f"\rModel indiriliyor: %{percent:5.1f}", end="", flush=True)


def select_embedding_model(manager: FoundryLocalManager):
    model = manager.catalog.get_model(REQUESTED_MODEL_ALIAS)
    if model is not None:
        return model

    embedding_models = [
        candidate
        for candidate in manager.catalog.list_models()
        if candidate.info.task == "embeddings"
        or "embedding" in (candidate.capabilities or "").lower()
    ]

    print(
        f"'{REQUESTED_MODEL_ALIAS}' alias'ı katalogda bulunamadı. "
        "Mevcut embedding modelleri:"
    )
    for candidate in embedding_models:
        print(f"  - {candidate.alias} ({candidate.id})")

    preferred_model = next(
        (
            candidate
            for candidate in embedding_models
            if candidate.alias == PREFERRED_MODEL_ALIAS
        ),
        None,
    )
    if preferred_model is not None:
        return preferred_model

    qwen_06b_model = next(
        (
            candidate
            for candidate in embedding_models
            if "qwen3" in candidate.alias.lower()
            and "0.6b" in candidate.alias.lower()
        ),
        None,
    )
    if qwen_06b_model is not None:
        return qwen_06b_model

    available_aliases = ", ".join(
        candidate.alias for candidate in embedding_models
    ) or "yok"
    raise RuntimeError(
        "Uygun Qwen3 0.6B embedding modeli bulunamadı. "
        f"Embedding model alias'ları: {available_aliases}"
    )


def main() -> None:
    print("Foundry Local SDK başlatılıyor...")
    FoundryLocalManager.initialize(Configuration(app_name="local_rag_test"))
    manager = FoundryLocalManager.instance
    print("Foundry Local başarıyla başlatıldı.")

    model = select_embedding_model(manager)
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

        embedding_client = model.get_embedding_client()
        response = embedding_client.generate_embeddings(SENTENCES)
        embeddings = sorted(response.data, key=lambda item: item.index)

        if len(embeddings) != len(SENTENCES):
            raise RuntimeError(
                f"{len(SENTENCES)} embedding bekleniyordu, "
                f"{len(embeddings)} embedding döndü."
            )

        for sentence, item in zip(SENTENCES, embeddings):
            vector = item.embedding
            if not vector:
                raise RuntimeError(f"Boş embedding döndü: {sentence}")

            print()
            print(f"Cümle: {sentence}")
            print(f"Vektör boyutu: {len(vector)}")
            print(f"İlk 5 değer: {vector[:5]}")
    finally:
        if model_loaded:
            model.unload()
            print()
            print(
                "Model çalışma belleğinden çıkarıldı; "
                "disk önbelleği korundu."
            )


if __name__ == "__main__":
    main()
