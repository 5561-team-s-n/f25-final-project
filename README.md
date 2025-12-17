# Setup Environment
First, download the code zip (may be a little faster since it doesn't have .git history) or `git clone` the repository. 
NOTE that no extra downloads are necessary since the checkpoints we trained from scratch are present in the pretrained/ folder as Git LFS objects. 
## Downloading Dependencies
### Option A (Slower)
You can download packages to a local venv with:
```bash
python3 -m venv .venv
source .venv/bin/activate # if on Windows, .\.venv\Scripts\Activate.ps1 
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
However, the whole install (including PyTorch w/ CUDA) is about 5GB, so this may take a bit.

### Option B (Faster)
There's a program called `uv` written in Rust that's much faster with setting up virtual environments (highly recommended!)
Install it and download packages with the following command:
```bash
## Install `uv`:
curl -LsSf https://astral.sh/uv/install.sh | sh # you could also do `brew install uv` on MacOS, or download through your OS's package manager 
## if on Windows, run the following instead of the above:
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
## At this point, you'll probably have to close and reopen your terminal for the `uv` command to work in your shell. Then:
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt --index-strategy unsafe-best-match # extra index-strategy flag is required since we use a custom cuda torch index url 
```
But if this doesn't seem to work on your machine, you can use Option A (which is tried and true). 

# Running the Code

You can reproduce our results by simply running `python pipeline_demo.py`.  
Notes: 
- By default, we have picked a sample foreground and background from the `samples/` folder to showcase the functionality, but you can specify other images with the command line arguments `--fg <path/to/image> --bg <path/to/image>`. 
- Results will end up in `/samples/output`, which may be changed with `--out-dir <path>`.  
- Pass `--debug` to get intermediate results (such as depthmaps and mattes) in `/samples/output/debug`. 
- By default, a modified trimap-free Matteformer trained from scratch is used for higher quality mattes. Pass `--disable_matteformer` to skip the Matteformer refinement step and only use our previous, lower quality model. (Note that the Matteformer code requires CUDA, so if this is not available, the disable flag will fix the issue).
- Depthmaps can take ~1-2 minutes to generate, so there is code to cache them by default (into `/samples/output/cache`). This speeds up inference on successive runs. Pass `--disable_cache` to prevent this.  
- See bottom of `pipeline_demo.py` for further command line arguments

As mentioned above, if running the program on a new foreground and background, everything should take about 1-2 minutes to complete. But thereafter depthmaps will be cached and runs should only take about 10 seconds.    

Here are some runs that re-generate results in our paper: 
```sh
python pipeline_demo.py # Hooked up by default to generate pineapple behind pumpkins 
python pipeline_demo.py --fg ./samples/fg2.png --bg ./samples/bg6.png --depth_shift 0.0 # Dog in front of girl
python pipeline_demo.py --fg ./samples/fg6.png --bg ./samples/bg7.jpg --depth_shift 1.0 # Flowers
python pipeline_demo.py --fg ./samples/fg2.png --bg ./samples/bg2.jpg --depth_shift 1.5 # Dog underwater w/ smooth composite
python pipeline_demo.py --fg ./samples/fg8.png --bg ./samples/bg4.png --depth_shift 1.0 # Horse in bushes
# Pass --debug to get intermediate pipeline steps too. 
```

# Acknowledgments
- Some utilities for training / losses of the image harmonization step use code from DCCF by [Xue et. al.](https://github.com/rockeyben/DCCF) which is present in the `iharm/` folder. 
- Some Matteformer utilities present in `utils/` and `networks/` comes from [Park et. al.](https://github.com/webtoon/matteformer) 