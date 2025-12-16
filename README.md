# Setup Environment
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
There's a program called `uv` that's extremely fast with python package management (highly recommended!)
Install it and download packages with the following command:
```bash
## Install `uv`:
curl -LsSf https://astral.sh/uv/install.sh | sh # you could also do `brew install uv` on MacOS, or download through your OS's package manager 
## if on Windows, run the following instead of the above:
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
## At this point, you'll probably have to close and reopen your terminal for the command to take effect. Then:
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt --index-strategy unsafe-best-match # flag is required since 
```
But if this doesn't work, you can use Option A (which is tried and true). 
# Running the Code

Use like `python pipeline_demo.py --debug --fg <./samples/image> --bg <./samples/image>`
See bottom of pipeline_demo.py for all command line arguments

TODO:
- [ ] Writing report -- datasets used were AIM500 for training mattenet, iHarmony for training image harmonizer. Both were used as foregrounds/backgrounds in samples
- [ ] We need metrics and to be able to test this on a bunch of different images
- [ ] README needs to be created, and we need to test that requirements.txt works properly
- [ ] (John) Add timing code with tqdm, deal with python warnings
- [ ] (John) Some way of specifying the amount to shift the foreground in the composite on the commandline? Currently you have to do it manually in occlusion_composite.py
