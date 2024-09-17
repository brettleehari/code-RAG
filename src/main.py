import sys
import platform

def print_banner():
    print("Welcome to Code Assistant Based on Llama-Index OSS Framework!")
    print("--------------------------")

def check_python_version(required_version):
    if sys.version_info < required_version:
        print(f"Python version {required_version} or higher is required.")
        return False
    return True

def check_pip_modules(required_modules):
    installed_modules = [module.split('==')[0] for module in sys.modules.keys()]
    missing_modules = [module for module in required_modules if module not in installed_modules]
    if missing_modules:
        print("The following pip modules are required but not installed:")
        for module in missing_modules:
            print(f"- {module}")
        return False
    return True

def main():
    print_banner()

    # Check Python version
    required_python_version = (3, 7)  # Example: Python 3.7 or higher
    if not check_python_version(required_python_version):
        return

    # Check required pip modules
    required_pip_modules = ["numpy", "pandas", "matplotlib"]  # Example: Required pip modules
    if not check_pip_modules(required_pip_modules):
        return
    
    print("Python version and pip modules are ready!")

    # Call the backend application

if __name__ == "__main__":
    main()