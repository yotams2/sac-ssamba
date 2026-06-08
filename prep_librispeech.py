import os
import json
import glob

def create_json(split_dirs, out_file):
    base_dir = "/storage/data/LibriSpeech/"
    data = []
    
    for split in split_dirs:
        split_path = os.path.join(base_dir, split)
        # LibriSpeech audio files are typically nested like: split/speaker/chapter/file.flac
        flac_files = glob.glob(os.path.join(split_path, "**", "*.flac"), recursive=True)
        
        for f in flac_files:
            data.append({
                "wav": f,
                "labels": "/m/09x0r"  # Using the AudioSet 'Speech' label MID to avoid label lookup errors
            })
            
    print(f"Found {len(data)} files for {out_file}")
    with open(out_file, "w") as f:
        json.dump({"data": data}, f, indent=4)

if __name__ == "__main__":
    train_splits = ["train-clean-100", "train-clean-360"]
    eval_splits = ["dev-clean"]
    
    create_json(train_splits, "librispeech_train.json")
    create_json(eval_splits, "librispeech_eval.json")
    print("Done!")
