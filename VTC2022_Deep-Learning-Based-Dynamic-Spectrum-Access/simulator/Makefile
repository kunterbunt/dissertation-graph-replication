.PHONY: install simulation

use-x86:
	unlink Pipfile
	ln -s Pipfile_x86 Pipfile
	unlink Pipfile.lock
	ln -s Pipfile_x86.lock Pipfile.lock
	pipenv update

use-mac-arm:
	unlink Pipfile
	ln -s Pipfile_Mac_ARM Pipfile
	unlink Pipfile.lock
	ln -s Pipfile_Mac_ARM.lock Pipfile.lock
	pipenv update

# make git ignore local changes to Pipfile and Pipfile.lock
# from https://stackoverflow.com/questions/1274057/how-do-i-make-git-forget-about-a-file-that-was-tracked-but-is-now-in-gitignore
dont-track-pipfile:
	git update-index --skip-worktree Pipfile Pipfile.lock

# make git respect local changes to Pipfile and Pipfile.lock
# from https://stackoverflow.com/questions/1274057/how-do-i-make-git-forget-about-a-file-that-was-tracked-but-is-now-in-gitignore
do-track-pipfile:
	git update-index --no-skip-worktree Pipfile Pipfile.lock

install:
	pipenv install --dev

ARGS=

simulation:
	pipenv run python main_rnn_roc.py ${ARGS}

tests:
	pipenv run pytest
