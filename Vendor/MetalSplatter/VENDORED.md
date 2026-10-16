# Vendored MetalSplatter

- Upstream: https://github.com/scier/MetalSplatter (MIT, see `LICENSE`)
- Version: v1.0.1, commit `71ff248e3016ac43c0a9271e322538421b28c360`
- Included: `PLYIO`, `SplatIO`, `MetalSplatter` targets (sources and shaders only).
  Tests, `SampleApp`, `SampleBoxRenderer` and `SplatConverter` are left out.

## Why vendored

Upstream converts every splat color to linear (`pow(c, 2.2)`) and blends in linear space
(sRGB render target). 3DGS trainers, including this one, alpha-composite the sRGB-encoded
colors directly, and a trained model is only correct under that compositing. Linear blending
brightens every pixel where colors mix (edges, thin structures, semi-transparent splats),
so trained scenes looked washed out in the viewer. Measured with the CLI's former `color-audit`
command on a capture dataset: luma +0.03–0.04 overall and +0.04–0.06 in dark tones compared with
the training renderer.

## Local patches

All are marked `LOCAL PATCH` in the source.

1. `MetalSplatter/Resources/SplatProcessing.h`: added the function constant
   `kLinearColorBlending` and `colorForBlending()`.
2. `MetalSplatter/Resources/SplatProcessing.metal`: `splatVertex` returns the sRGB-encoded
   color instead of linearizing it.
3. `MetalSplatter/Resources/{Single,Multi}StageRenderPath.metal`: the vertex shaders apply
   `colorForBlending()` to the color returned by `splatVertex`.
4. `MetalSplatter/Sources/SplatRenderer.swift`: the splat vertex functions are created with
   `kLinearColorBlending = colorFormat.isSRGB`.

5. `MetalSplatter/Sources/SplatSorter.swift`: the CPU sort runs only when it is needed.
   - `updateCameraPose` requests a sort only when the pose changed. Upstream requested one on
     every call, and `SplatRenderer.render` calls it every frame, so all splats were re-sorted
     back to back while the camera was still.
   - The sort loop exits when no sort is pending, and clears `sortLoopRunning` under the same
     lock as that decision. Upstream kept the loop alive while any chunk existed, waking every
     1 ms.
   - `invalidateAllBuffers` restarts the loop, and `addSortCompletionHandler` requests a sort,
     so `afterNextSort` still fires while the camera is still.

   With a still camera the sorter no longer runs and process CPU drops to near idle.

6. Stochastic transparency (`SplatRenderer.transparencyMode`), adapted from SuperSplat's
   "Stochastic Alpha" (PlayCanvas, MIT):
   - `MetalSplatter/Resources/StochasticRenderPath.metal` (new): a vertex shader that reads the
     sorted index buffer in reverse (roughly front to back, so the depth test rejects hidden
     fragments early), a fragment shader that keeps each fragment with probability alpha
     (thresholds stratified over 2x2 pixel quads, hashed per quad and splat) and writes it
     opaque, and a resolve pass that averages quads and interpolates between quad centres.
     It uses no half-typed varyings: mixed with a 32-bit `[[flat]]` varying they arrived as zero
     on an M-series Mac unless Metal shader validation was on.
   - `MetalSplatter/Sources/SplatRenderer.swift`: `TransparencyMode` (`.sorted` upstream
     behaviour, `.stochastic`, `.movement` = stochastic while the camera moves and sorted
     blending once it is still and sorted for that pose), `stochasticQuadResolve`,
     `stochasticResortAngle`, `lastRenderWasStochastic`, the stochastic pipeline and passes,
     and `renderEncoder(clearDepth:)`. The stochastic modes sort only when the camera comes to
     rest or has turned `stochasticResortAngle` (45 degrees) since the last sort; reverse-Z
     projections are detected from the projection matrix for the depth test. Without a depth
     attachment renders fall back to `.sorted`.
   - `MetalSplatter/Sources/SplatSorter.swift`: index buffers remember the pose they were sorted
     for (`tryObtainSortedIndicesAndPose`), so `.movement` blends only an exact sort.

   While orbiting, CPU sorts and process CPU drop to a small fraction and GPU time per frame falls.
   At a still pose the quad resolve brings the image much closer to the sorted blend.

7. `MetalSplatter/Sources/SplatRenderer.swift`: `lastRenderWasFinal` tells whether the most
   recent render is the final image for its pose (sorted blending with indices sorted for exactly
   that pose, or the stochastic mode, whose pattern does not depend on the draw order). The app's
   viewer (`SplatViewerKit`) keeps its `MTKView` paused once a final frame is on screen and wakes
   it only when the camera, a layer toggle or the drawable size changes.
   Upstream's sample redraws every display frame, so a still viewer kept the GPU busy.

   With a still camera the viewer stops rendering and the GPU goes idle. After an orbit stops it
   renders a few more frames (until the sort for the resting pose lands) and pauses.

8. `MetalSplatter/Sources/SplatRenderer.swift`: `composeOverExistingContents` draws the splats over
   what is already in the color and depth targets instead of clearing them, so a viewer can render
   an opaque mesh first and show the splats around it. The pass loads color and depth, the splats
   are depth-tested against the existing depth (`lessEqual`, or `greaterEqual` for reverse-Z) and
   do not write depth, and blending is unchanged, so splats in front of the mesh blend over it and
   splats behind it are hidden. Only the sorted single-stage path supports it: while it is set the
   renderer uses sorted blending and the single-stage pipeline even where the multi-stage one would
   be used (the single-stage builders' asserts allow that).

9. `MetalSplatter/Sources/EncodedSplatPoint.swift`: `worldPosition`, `linearOpacity` and `covariance`
   read a stored splat back on the CPU (the fields are internal upstream). The app's viewer uses them to
   find the surface under a tap for its measuring tool without keeping a second copy of the splats.

10. `MetalSplatter/Resources/SplatProcessing.metal`: `calcCovariance2D` and `decomposeCovariance` are
    rewritten from the math:
    - `calcCovariance2D` follows EWA splatting (Zwicker et al. 2001): the view-space covariance
      R Σ Rᵀ is projected with the two rows of the perspective Jacobian.
    - `decomposeCovariance` uses the closed-form eigenvalues of a symmetric 2x2 matrix,
      (a + d) / 2 ± sqrt(((a - d) / 2)² + b²). Upstream took the square root of `mean² - det`,
      which cancels on large, nearly round splats.
    - The constants are upstream's: the Jacobian is evaluated with x/z and y/z clamped to
      1.3 × tan(half FOV), the diagonal is dilated by 0.3 px², and the eigenvalue half-gap is floored
      at 0.1 px². The footprints are unchanged.

    The 2D covariance matches upstream's to float precision, and against a double-precision
    reference the axis error is smaller than upstream's, most of all in the worst case.

Result of patches 1–4: the blend space follows the render target format. An sRGB target keeps upstream
behaviour (linear blending). A non-sRGB target such as `.bgra8Unorm` blends sRGB-encoded
colors, which matches training. Our viewers use `.bgra8Unorm`.

When updating to a newer upstream version, reapply these patches. If upstream gains an
equivalent option, drop the vendored copy and depend on upstream again.
