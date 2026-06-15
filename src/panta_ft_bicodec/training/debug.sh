module load arch/a100
source ../../../.venv/bin/activate

gdb python core-python-2541178-11 <<EOF
bt full
quit
EOF
