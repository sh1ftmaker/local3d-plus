#!/usr/bin/env bash
# Generate multi-subject reference sheets with Codex's image tool (one image per call), copy them into test_inputs/sheets.
set -u
OUT=/bigdata/agent-tom/projects/trellis/test_inputs/sheets
STYLE="Each subject fully visible and complete, with a wide empty margin around every subject and around the image edges; subjects must not overlap, touch or be cropped. Front three-quarter view, detailed 3D game-asset style render with PBR materials, soft even studio lighting, no cast shadows, plain uniform light grey background, no ground plane, no text, no labels, no frames, no grid lines."
gen() {  # name, subject description
  local name=$1 desc=$2
  [ -f "$OUT/$name.png" ] && { echo "skip $name"; return; }
  # workspace-write: the read-only sandbox blocks Codex's image tool
  local out; out=$(cd "$OUT" && timeout 900 codex exec --skip-git-repo-check -s workspace-write \
    "Generate exactly ONE image with your image generation tool (never more than one) and reply with only the generated file's path. Image: a wide landscape reference sheet showing $desc $STYLE" 2>&1)
  local f; f=$(grep -o '/[^ ]*generated_images/[^ ]*\.png' <<<"$out" | tail -1)
  if [ -n "$f" ] && [ -f "$f" ]; then cp "$f" "$OUT/$name.png"; echo "ok $name"; else echo "FAILED $name"; fi
}
gen sheet02_steampunk_engineers "FOUR different full-body steampunk engineer characters standing in one row, relaxed A-pose: (1) a bearded dwarf mechanic with brass goggles, leather apron and a giant wrench, (2) a female airship engineer in a corset vest with a mechanical arm and tool belt, (3) a tall gaunt inventor in a long coat with a backpack boiler and copper pipes, (4) a young apprentice with a clockwork owl on the shoulder and oversized gloves."
gen sheet03_wizards "FOUR different full-body wizards and mages standing in one row: (1) an old grey-bearded wizard in a starry blue robe and pointed hat with a gnarled staff, (2) a dark sorceress in black and purple layered robes with a floating spellbook, (3) a desert alchemist in sand-colored wraps with potion bottles on a bandolier, (4) a druid in moss-green robes with antler headdress and a wooden staff with leaves."
gen sheet04_adventurers "FOUR different full-body fantasy adventurers standing in one row: (1) a hooded human ranger with a cloak and quiver, (2) a heavily armored orc warrior with a battle axe, (3) a halfling bard with a lute and feathered hat, (4) a lizardfolk monk in simple wraps holding a staff."
gen sheet05_city_infrastructure "SIX separate city infrastructure objects arranged in two rows of three: an ornate cast-iron street lamp, a red fire hydrant, a covered bus stop shelter with bench, a wooden water tower on legs, a traffic light on a pole, and a stone and steel arch bridge segment."
gen sheet06_trees "SIX separate stylized trees and plants arranged in two rows of three: a gnarled ancient oak, a tall pine, a weeping willow, a cherry blossom tree, a twisted bonsai in a pot, and a giant glowing fantasy mushroom tree; each tree complete with its full canopy and trunk base."
gen sheet07_towers_landmarks "FOUR separate landmark buildings in one row: a gothic clock tower with a large ornate clock face, a stone lighthouse, a Victorian brick train station with a clock, and a tall wizard spire with spiral balconies; each building complete from base to rooftop."
gen sheet08_steampunk_machines "SIX separate steampunk machines and props arranged in two rows of three: a small airship with propellers, a brass steam locomotive engine, a clockwork mechanical spider, a telescope on a tripod, an ornate gear-driven grandfather clock, and a steam-powered motorcycle."
gen sheet09_fantasy_homes "FOUR separate fantasy homes in one row: a red-capped mushroom cottage with round windows, a treehouse built around a thick trunk with rope ladder, a hobbit-style hillside cottage with a round door, and a crooked witch's cottage with a leaning chimney; each home complete as an isolated model."
gen sheet10_fantasy_places "FOUR separate miniature fantasy diorama places in one row, each on its own small base: a floating sky island with a waterfall and a tiny temple, a castle on a rocky cliff, a cozy medieval market square with stalls, and a forest shrine with torii gate and stone lanterns."
tm-notify "Codex sheets done" "$(ls $OUT | wc -l) sheets in test_inputs/sheets" --tag sheets
