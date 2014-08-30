#include <stdio.h>


int can_print_it(char ch) {
    return isalpha(ch) || isblank(ch);
}

void print_letters(char arg[]) {
    int i = 0;

    for(i = 0; arg[i] != '\0'; i++) {
        char ch = arg[i];

        if(can_print_it(ch)) {
            printf("'%c' == %d ", ch, ch);
        }
    }

    printf("\n");
}

void print_arguments(int argc, char *argv[]) {
    int i = 0;
    for(i = 0; i < argc; i++) {
        print_letters(argv[i]);
    }
}


int main(int argc, char *argv[]) {

	print_arguments(argc, argv);

    int a = 1;
    int b = 2;
    char _string[] = "string";
    int areas[] = {10, 12, 13, 14, 20};
    int numbers[4] = {0, 1};

    printf("a: %d, b: %d \n", a, b);
    printf("%s \n", _string);
 	printf("The number of ints in areas: %ld\n",
           sizeof(areas) / sizeof(int));
 	printf("numbers: %d %d %d %d\n",
            numbers[0], numbers[1],
            numbers[2], numbers[3]);

 	int i = 0;
    for(i = 1; i < argc; i++) {
        printf("arg %d: %s\n", i, argv[i]);
    }

    return 0;
}
// make hworld && ./hworld
// CFLAGS="-Wall" make hworld && ./hworld
// valgrind ./hworld
