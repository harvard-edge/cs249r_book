echo '$ false | cat; echo $?'; false | cat; echo $?
echo '$ set -o pipefail; false | cat; echo $?'; ( set -o pipefail; false | cat; echo $? )
echo "\$ sh -c 'exit 256'; echo \$?"; sh -c 'exit 256'; echo $?
echo "\$ sh -c 'kill -9 \$\$'; echo \$?"; sh -c 'kill -9 $$'; echo $?
