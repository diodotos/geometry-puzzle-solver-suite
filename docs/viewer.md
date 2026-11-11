# Viewer: input mapping and intended behaviour

`gpss-viewer` (crate `crates/gpss-viewer`, ADR 0010) draws a generic `AxBxC` cuboid from the backend
`CuboidState` with eframe/egui and wgpu. All input goes through `gpss_viz::controller::Controller`, which
the desktop app and the headless recorder share, so the scenarios below exercise the same code a user does.

## Running

```bash
cargo run --release -p gpss-viewer -- --dims 3x3x3 --scramble 25    # desktop window
cargo run --release -p gpss-viewer -- record crates/gpss-viewer/scenarios/*.scn --out target/viewer-videos
```

`record` needs a Vulkan/GL adapter (on a headless Linux box, Mesa's lavapipe from `mesa-vulkan-drivers` works)
and `ffmpeg` with libx264 and libass. It writes `<name>.mp4` (cursor drawn in: white = up, orange = primary
down, magenta = secondary down, yellow line back to the press point; captions = case and intended behaviour;
yellow lines = controller feedback; green/red = expectation verdicts; bottom line = status) and `<name>.log`.
`cargo test -p gpss-viewer` plays every scenario through the controller without rendering and checks the
expectations, so the mapping below is enforced in CI. The script format is documented in `src/scenario.rs`.

## Input mapping

| Input | Behaviour |
|---|---|
| Primary press on a sticker, then drag | Once the pointer is 8 px from the press and the drag (projected onto the sticker's plane) is dominated 1.5:1 by one in-plane direction, the layer through the sticker that moves that way turns one quarter (half on axes whose cross-section is not square). One press gives at most one turn. |
| Press and release on a sticker without moving 8 px | Click: no move. |
| Drag that never becomes dominant (diagonal) | No move; feedback "diagonal drag". |
| Primary press off the cube, or secondary press anywhere, then drag | Orbit: 0.01 rad/px; yaw wraps, pitch is clamped to +/-85 degrees. Never turns a layer. |
| Wheel | Zoom, clamped to 0.6x-3x the framing distance. Allowed during a drag. |
| Keys `U R F D L B` (Shift = prime), notation box | Queue those turns. Invalid notation (including quarter turns on half-turn-only axes) is rejected whole. |
| Ctrl+Z / Undo | Drop pending solver moves, then undo the last shown or queued move (an animating move finishes, then plays backwards). |
| Scramble / setup / Reset | Jump straight to the new state; queued turns and running solves are discarded. |
| S / Solve | Solve the state the queue will end in (two-phase on 3x3x3, IDA* otherwise) on a worker thread, then play the solution. Ignored while a solve runs; "already solved" when solved. An answer for a state that changed meanwhile is dropped. |
| Any user turn during solver playback | Remaining solver moves are dropped; the user's turn plays. |
| Home view | Camera back to the starting view (puzzle untouched). |
| More than 64 pending moves | Further moves are rejected. |

Picking ignores the animation in progress: it uses the stickers' resting positions, so during a turn the pointer picks the backend slot whose resting position is under it.

## Scenarios

Each case below is one captioned segment of the recorded videos; the second column is the caption's
intended behaviour, checked by the `expect` lines in the script and by watching the video.

### `01_basic_turns.scn`

| Case | Intended behaviour |
|---|---|
| 1.1 Drag F right column up | R: right layer turns clockwise seen from R (front column goes up) |
| 1.2 Drag F right column down | R': the same layer the other way |
| 1.3 Drag F left column down | L: left layer clockwise seen from L (shown as its R-axis name 3R') |
| 1.4 Drag F top row left | U: top layer clockwise seen from above |
| 1.5 Drag F bottom row right | D: bottom layer clockwise seen from below (3U') |
| 1.6 Drag U front row right | F: front layer clockwise seen from the front |
| 1.7 Drag R front column down | F again, grabbed on the R face: front layer clockwise |
| 1.8 Drag U back row left | B: back layer clockwise seen from behind (3F') |
| 1.9 Drag F middle column up | 2R: middle slice turned like R |
| 1.10 Drag F middle row left | 2U: equator slice turned like U |
| 1.11 Drag U middle row right | 2F: standing slice turned like F |
| 1.12 Notation input "R U R' U'" | Four queued turns animate one after another |
| 1.13 Notation input "R2 U2" | Half turns animate as one 180 degree turn each |

### `02_camera.scn`

| Case | Intended behaviour |
|---|---|
| 2.1 Drag empty space right | Orbit: the cube spins about the vertical; no turn |
| 2.2 Drag empty space up | Orbit: view tilts so the bottom (D, yellow) comes into view; no turn |
| 2.3 Keep tilting far past the pole | Pitch is clamped at 85 degrees; the view never flips over |
| 2.4 Right-drag across the cube | A secondary-button drag orbits even when it starts on a sticker |
| 2.5 Wheel up, then down | Zoom in, then out |
| 2.6 Wheel far past both limits | Zoom is clamped between 0.6x and 3x the framing distance |

### `03_compositions.scn`

| Case | Intended behaviour |
|---|---|
| 3.1 Orbit to look from below, then drag a D sticker | Turns follow the grabbed sticker in any view: D right column towards the front = R |
| 3.2 Orbit to look from the back-left, then drag B | Dragging the B face's top row (left on screen) is B-face U: top layer clockwise |
| 3.3 Orbit back to the front | Dragging right by the same distance orbits back to the starting view |
| 3.4 Two drags then two undos | R, U by drag; undo, undo plays them backwards to solved |
| 3.5 Notation then a drag while it plays | "R U" queued, a drag adds F behind them: R U F |
| 3.6 Orbit while a turn animates | The turn finishes normally while the view moves |
| 3.7 Scramble then solve | Two-phase solution plays back to solved |
| 3.8 Undo after the solve | Undo takes back the last solver move |

### `04_conflicts.scn`

| Case | Intended behaviour |
|---|---|
| 4.1 Click a sticker without moving | No move; feedback says click |
| 4.2 Drag 5 px (under the 8 px threshold) | Still a click: no move |
| 4.3 Exact diagonal drag on F | Ambiguous: no move, feedback says diagonal |
| 4.4 Drag up, then sideways, in one press | One press = at most one turn: R, then the sideways part is ignored |
| 4.5 Drag from a sticker far off the cube | Turn R fires as usual; leaving the cube neither orbits nor adds a turn |
| 4.6 Drag from empty space across the cube | Orbit only: a press that starts off the cube never turns |
| 4.7 Right button pressed during a sticker drag | Second button ignored; the primary drag still turns R |
| 4.8 Release with no button down | Ignored |
| 4.9 Drag a new turn while one animates | Queued: R then U, in order |
| 4.10 Undo during the animation of the same move | R finishes, then plays backwards: back to solved |
| 4.11 Undo with nothing to undo | Ignored |
| 4.12 User drag during solver playback | Remaining solver moves are dropped, the user's turn plays, cube stays unsolved |
| 4.13 Undo during solver playback | Remaining solver moves are dropped and the last shown move is undone |
| 4.14 Solve pressed twice | Second press ignored while the first solve runs; one solution plays |
| 4.15 Move while the solver is thinking | The late answer is for an old state: dropped; the cube keeps the user's move |
| 4.16 Solve when already solved | No solve runs |
| 4.17 Scramble while a sequence is playing | Jump straight to the scramble; queued turns are discarded |
| 4.18 Invalid notation | Rejected, nothing moves |
| 4.19 Overfill the queue | Only 64 moves fit in the queue: 64 x R play (= identity), the last 6 are rejected |
| 4.20 Zoom with the wheel during a sticker drag | The wheel zooms; the drag still produces its turn R |

### `05_cuboids.scn`

| Case | Intended behaviour |
|---|---|
| 5.1 Drag F right column up on 3x3x2 | R-axis layers are 3x2: a quarter-turn gesture becomes the legal half turn R2 |
| 5.2 Drag F top row left | U layer is 3x3: an ordinary quarter turn U |
| 5.3 Drag U front row right | F-axis layers are 3x2: half turn F2 |
| 5.4 Notation "R" on 3x3x2 | Rejected: a quarter R is not a legal move here |
| 5.5 Scramble and solve 3x3x2 | IDA* solution plays back to solved |

### `06_big_cube.scn`

| Case | Intended behaviour |
|---|---|
| 6.1 Drag F second column from the right up | 2R: the inner layer next to R |
| 6.2 Drag F second row from the bottom right | 3U': the inner layer next to D, turned like D (= 3U') |
| 6.3 Scramble 5 and solve | IDA* finds a short solution and plays it back |
