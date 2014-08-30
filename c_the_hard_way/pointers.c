
#include <stdio.h>

int main(int argc, char * argv[]) {
    char *names[] = {
        "Alan", "Frank",
        "Mary", "John", "Lisa"
    };
    int i = 0;
    //wtf is goin on here, need some beard to figure out
    int count = sizeof(names) / sizeof(*names);
    printf("%p\n", *names);
    char **cur_name = names;

    printf("%d\n", count);

    for(i = 0; i < count; i++) {
        printf("name: %s\n", *(cur_name+i));
    }

    return 0;
}
/**
type *ptr -> "a pointer of type named ptr"
*ptr -> "the value of whatever ptr is pointed at"
*(ptr + i) -> "the value of (whatever ptr is pointed at plus i)"
&thing -> "the address of thing"
type *ptr = &thing -> "a pointer of type named ptr set to the address of thing"
ptr++ -> "increment where ptr points"
**/
