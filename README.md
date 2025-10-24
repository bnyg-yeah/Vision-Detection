For prerecorded pipeline:

    Animal ML Commands

    To enter the virtual python environment:
    ls ~/venvs
    source ~/venvs/speciesnet/bin/activate

    To run ffmpeg:
    ffmpeg -i OriginalVideoName.mp4 -vf fps=3 -qscale:v 2 -start_number 0 frames/FrameFolderName/ts_%010d.jpg

    To run speciesnet:
    python -m speciesnet.scripts.run_model --folders "/Users/brightonyoung/ML-Projects/Vision-ML/frames/PreciseTrail2" --predictions_json "/Users/brightonyoung/ML-Projects/Vision-ML/output/precise_output2.json" --country USA --admin1_region AL

    To run the annotater:
    1. customize JSON_PATH, VIDEO_IN, VIDEO_OUT, JSON_FPS correctly in the file
        1. JSON_FPS = (the number of frames / the length of video)
    2. python annotateVideos.py

    To leave the virtual python environment:
    deactivate

