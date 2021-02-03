shadertoy-render
================

**Update 2021:**
TODO - I will update this script in next few months...
Sound support will be added, better "user friendly" and rewriting this code to make it better, with adding Windows support.

**Windows OS support for now :**
1. install `Cygwin`
2. from Cygwin packages UI install `python` latest python38 and `python38-devel` and `python38-numpy` and `libfreetype-devel`
3. 
```
> python -m pip install --upgrade pip
> pip install numpy

**What is it**: creating video from shaders on Shadertoy. Fork from `original <https://github.com/alexjc/shadertoy-render>`_, source code edited

**Changes**:

1. added texture support, textures in **<0.png to 3.png> bind to iTexture<0-3>**
2. added FrameBuffers, same to Buffer<A-D> on Shadertoy, **file name Buf<0-3>.glsl**, bind to *iChannel<0-3>* and *u_channel<0-3>*
3. added encoding to \*.mov (frames without compression), \*.webm (v8 codec), both format **support RGBA** (video with alpha), added *--bitrate* option to set video bitrate(format 1M 2M..etc)
4. fixed iTime(start from 0 on shader launch), iFrame work, iTimeDelta, and other

**Warning**

Many shaders(even top rated) on Shadertoy use clamp(1,0,-1)/pow(1,-1)/(0/0)/...etc, that work in not same way(have not same result) in OpenGL and webbrowser Angle/GLES, black screen(or other random "results") because of this. Also remember to set Alpha in main.glsl when recording rgba video. interpolation of fbo is linear 

**Example**

each shader(buffer) use static bindings iChannel0->Buf0.glsl, iChannel1->Buf1.glsl, iChannel2->Buf2.glsl, iChannel3->Buf3.glsl, also **added renamed copy of each channel** *sampler2D u_channel<0-3>*, to rebind inside of .glsl shader(using define)

if you need "change" channel order for shader, use in .glsl file (example set BufA as BufC, and BufC as Texture0(0.png file))

	#define iChannel0 u_channel3
	
	#define iChannel3 iTexture0
	
	
use same way to bind iTexture<0-3> as iChannel<0-3> *#define iChannel0 iTexture0*

**check example folder**, command to encode example(example use 3 buffers and one texture):

	> cd example
	
	> python3 ../shadertoy-render.py --output 3.mp4 --size=800x450 --rate=60 --duration=20.0 --bitrate=5M main_img.glsl

to record \*.mov or \*.webm just change output file to 3.webm or 3.mov


**Example 1** `shader src <https://www.shadertoy.com/view/MdGGzG>`_ webm video recorded with RGBA and bufA `video link <https://danilw.github.io/GLSL-howto/shadertoy-render/1.webm>`_

**Example 2** `shader src <https://www.shadertoy.com/view/ltGBRD>`_ mp4 60fps/sec use BufA<->BufB cross reading
`video link <https://danilw.github.io/GLSL-howto/shadertoy-render/2.mp4>`_

**Example 3** `shader src <https://www.shadertoy.com/view/3dl3z7>`_ mp4 60fps/sec cross and self reading, this shader used in example folder
`video link <https://danilw.github.io/GLSL-howto/shadertoy-render/3.mp4>`_
