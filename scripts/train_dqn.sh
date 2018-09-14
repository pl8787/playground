set -e
set -x
export CUDA_VISIBLE_DEVICES=0
P1="train::agents.DQNAgent++./model/conv3x256.json"
P2="test::agents.SimpleAgent"
P3="test::agents.SimpleAgent"
P4="test::agents.SimpleAgent"
pom_tf_battle \
    --agents=$P1","$P2","$P3","$P4 \
    --config=PommeTeamCompetition-v0 \
    --model_save_dir="./dqn_model/model_team_conv3x256"
    # --render #\
    #--render_mode rgb_array \
    #--record_json_dir="../records/json/"
