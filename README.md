# Running the Code

Use like `python pipeline_demo.py --debug --fg <./samples/image> --bg <./samples/image>`
See bottom of pipeline_demo.py for all command line arguments

TODO:
- [ ] Writing report -- datasets used were AIM500 for training mattenet, iHarmony for training image harmonizer. Both were used as foregrounds/backgrounds in samples
- [ ] We need metrics and to be able to test this on a bunch of different images
- [ ] README needs to be created, and we need to test that requirements.txt works properly
- [ ] (John) Add timing code with tqdm, deal with python warnings
- [ ] (John) Some way of specifying the amount to shift the foreground in the composite on the commandline? Currently you have to do it manually in occlusion_composite.py
