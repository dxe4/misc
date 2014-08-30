# include <stdio.h>
# include <assert.h>
# include <stdlib.h>
# include <string.h>

struct Person {
    char * name;
    int age;
    int height;
    int weight;
};

struct Thing {
	char * a_string;
	int an_int;
};

struct Thing * Thing_create(char * a_string, int an_int) { 
	struct Thing * thing = malloc(sizeof(struct Thing));
	assert(thing != NULL);

	thing->a_string = strdup(a_string);
	thing->an_int = an_int;
	return thing;
}

void Thing_destroy(struct Thing * thing) {
	assert(thing != NULL);
	free(thing->a_string);
	free(thing);
}

void Thing_print(struct Thing * thing) { 
	printf("%s\n", thing->a_string);
	printf("%d\n", thing->an_int);
}

struct Person * Person_create(char * name, int age, int height, int weight) {
    struct Person * who = malloc(sizeof(struct Person));
    assert(who != NULL);

    who->name = strdup(name);
    who->age = age;
    who->height = height;
    who->weight = weight;
    return who;
}

void Person_destroy(struct Person * who) {
    assert(who != NULL);

    free(who->name);
    free(who);
}

void Person_print(struct Person * who) {
    printf("Name: %s\n", who->name);
    printf("\tAge: %d\n", who->age);
    printf("\tHeight: %d\n", who->height);
    printf("\tWeight: %d\n", who->weight);
}

void people(){
	// make two people structures
    struct Person * joe = Person_create(
        "Joe Alex", 32, 64, 140);

    struct Person * frank = Person_create(
        "Frank Blank", 20, 72, 180);

    // print them out and where they are in memory
    printf("Joe is at memory location %p:\n", joe);
    Person_print(joe);

    printf("Frank is at memory location %p:\n", frank);
    Person_print(frank);

    // make everyone age 20 years and print them again
    joe->age += 20;
    joe->height -= 2;
    joe->weight += 40;
    Person_print(joe);

    frank->age += 20;
    frank->weight += 20;
    Person_print(frank);

    // destroy them both so we clean up
    Person_destroy(joe);
    Person_destroy(frank);
}

void things() {
	struct Thing * thing_a = Thing_create(
		"a_string", 99);
	Thing_print(thing_a);
	Thing_destroy(thing_a);
}

int main(int argc, char * argv[]) {
	people();
	things();
    return 0;
}
