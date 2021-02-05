#define iChannel0 iTexture0
#define iChannel1 iTexture1

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    vec2 ouv=uv*2.;
    uv=fract(uv*2.);
    vec3 col=vec3(0.);
    if(ouv.x<1.)
    	col = texture(iChannel0,uv).rgb;
    else
        col = texture(iChannel1,uv).rgb;
    fragColor = vec4(col,1.0);
}
