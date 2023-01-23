# find cookiecutter package in my conda environments
for env in $(conda env list | grep -v '^#' | awk '{print $1}'); do
    if [[ -d $(conda env list | grep -v '^#' | grep $env | awk '{print $2}')/lib/python3.7/site-packages/cookiecutter ]]; then
        echo "Found cookiecutter in $env"
        break
    fi
done
