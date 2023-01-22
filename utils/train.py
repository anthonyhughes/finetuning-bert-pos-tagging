def train(model, iterator, optimizer, criterion, training_steps: int) -> None:
    """

    :param model: nn to be trained
    :param iterator: dataset to be used as training
    :param optimizer:
    :param criterion:
    :param training_steps: stop at set step count
    :return:
    """
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        optimizer.zero_grad()
        logits, y, _ = model(x, y)  # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i % 10 == 0:
            print(f"step={i}, loss={loss.item}")

        if i == training_steps:
            break
