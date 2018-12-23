shadertoy-render
================

**What is it**: creating video from shaders on Shadertoy

**Changes**:

1. added texture support, textures in **<0.png to 3.png> bind to iTexture<0-3>**
2. added FrameBuffers, same to Buffer<A-D> on Shadertoy, **file name Buf<0-3>.glsl**, bind to iChannel<0-3>
3. added encoding to \*.mov (frames without compression), \*.webm (v8 codec), both format **support RGBA** (video with alpha), added *--bitrate* option to set video bitrate(format 1M 2M..etc)
4. fixed iTime(start from 0 on shader launch), iFrame work, iTimeDelta, and other

**Warning**

Many shaders(even top rated) on Shadertoy use clamp(1,0,-1)/pow(1,-1)/(0/0)/...etc, that work in not same way(have not same result) in OpenGL and webbrowser Angle/GLES, black screen(or other random "results") because of this. Also remember to set Alpha in main.glsl when recording rgba video. interpolation of fbo is linear 

**Example**

each shader(buffer) use static bindings iChannel0->Buf0.glsl, iChannel1->Buf1.glsl, iChannel2->Buf2.glsl, iChannel3->Buf3.glsl

if you need "change" order for shader, use in .glsl file (example set BufA as BufC)

	#define iChannel0 edit....
	
use same way to bind iTexture<0-3> as iChannel<0-3> *#define iChannel0 iTexture0*

**check example folder**, command to encode example:

	> python shadertoy-render.py ...


**Example 1** `shader src <https://www.shadertoy.com/view/MdGGzG>`_ webm video recorded with RGBA and bufA `video link <https://danilw.github.io/GLSL-howto/shadertoy-render/1.webm>`_

**Example 2** `shader src >https://www.shadertoy.com/view/ltGBRD>`_ mp4 60fps use BufA<->BufB cross 
`video link <https://danilw.github.io/GLSL-howto/shadertoy-render/2.mp4>`_

**Example 3** 
