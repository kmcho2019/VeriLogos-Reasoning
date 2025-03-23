def print_trainable_parameters(stage, model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"[{stage}]: Trainable params: {trainable_params} || All params: {all_param} || Trainable%: {100 * trainable_params / all_param}"
    )

