from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, text_list, tokenizer, block_size=512):
        self.examples = []
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens(
                {
                    "pad_token": (
                        tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"
                    )
                }
            )
            print("Pad Token:" + tokenizer.pad_token)
        for text in text_list:
            tokenized_text = tokenizer(
                text,
                truncation=True,
                max_length=block_size,
                padding="max_length",  # Add padding to make all sequences the same length
                return_tensors="pt",
            )
            self.examples.append(tokenized_text.input_ids.squeeze())

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
