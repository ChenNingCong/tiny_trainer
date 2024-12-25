import glob
import os
import subprocess
import shlex

def run_coverage_for_yaml():
    # Get all yaml files in current directory
    yaml_files = glob.glob("*.yaml") + glob.glob("*.yml")
    
    if not yaml_files:
        print("No YAML files found in current directory")
        return
    
    coverage_files = []
    print(yaml_files)
    for yaml_file in yaml_files:
        print(f"Running coverage for {yaml_file}")
        coverage_file = f'{yaml_file}..coverage'
        try:
            # Run coverage directly using command line
            new_env = os.environ.copy()
            new_env["WANDB_MODE"] = "offline"
            subprocess.run(shlex.split(f'coverage run --data-file={coverage_file} test_trainer.py hydra.output_subdir=null --config-name {yaml_file}'), 
            check=True, env=new_env)
            coverage_files.append(coverage_file)
            
        except subprocess.CalledProcessError as e:
            print(f"Error running coverage for {yaml_file}: {e}")
            continue
    
    if coverage_files:
        # Combine coverage data
        subprocess.run(['coverage', 'combine', '--keep'] + coverage_files)
        
        # Generate HTML report
        subprocess.run(['coverage', 'html'])
        print("Combined HTML coverage report generated in 'htmlcov' directory")
    else:
        print("No coverage data was generated")

if __name__ == "__main__":
    run_coverage_for_yaml()