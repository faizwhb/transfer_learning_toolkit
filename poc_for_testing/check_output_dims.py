from tools.models import make

my_densenet = make(model_name='densenet', num_classes=10)
print(my_densenet)
my_mobilenet = make(model_name='mobilenet', num_classes=10)
print(my_mobilenet)