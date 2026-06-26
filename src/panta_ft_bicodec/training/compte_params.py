from panta_ft_bicodec.model.bicodec_tokenizer import BiCodecTokenizer

model = BiCodecTokenizer()
total_param = 0

for param in model.model.decoder.parameters():
    total_param += param.numel()

for param in model.model.prenet.parameters():
    total_param += param.numel()


print(total_param / 1e6)