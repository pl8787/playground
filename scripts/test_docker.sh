set -e
set -x
P1="test::agents.SimpleAgent"
P2="test::agents.SimpleAgent"
# P3="test::agents.SimpleAgent"
P3="docker::pommerman/random-agent" 
# P4="test::agents.SimpleAgent" 
P4="docker::pommerman/simple-agent" 
pom_battle \
    --agents=$P1","$P2","$P3","$P4 \
    --config=PommeFFACompetition-v0 \
    --render \
    --render_mode rgb_array \
    --record_json_dir="../records/json/"

