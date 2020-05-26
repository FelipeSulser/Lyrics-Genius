import lyricsgenius


lg = lyricsgenius.lyricsgenius(network_path="../models/rhcp_model_res.h5", alphabet_path="../models/rhcp-alphabet.json", tokenizer_path="../models/rhcp-tokenizer.pkl")
print(lg.predict("This is the life "))