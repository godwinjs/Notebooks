Requirements

You can run Avatarify in two modes: locally and remotely.

To run Avatarify locally you need a CUDA-enabled (NVIDIA) video card. Otherwise it will fallback to the central processor and run very slowly. These are performance metrics for some hardware:

    GeForce GTX 1080 Ti: 33 frames per second
    GeForce GTX 1070: 15 frames per second
    GeForce GTX 950: 9 frames per second

You can also run Avatarify remotely on Google Colab (easy) or on a dedicated server with a GPU (harder). There are no special PC requirements for this mode, only a stable internet connection.

Of course, you also need a webcam!
Install
Download network weights

Download model's weights from here or here or here [228 MB, md5sum 8a45a24037871c045fbb8a6a8aa95ebc]
Linux

Linux uses v4l2loopback to create virtual camera.

    Download Miniconda Python 3.7 and install using command:
! bash Miniconda3-latest-Linux-x86_64.sh
