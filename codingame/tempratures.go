package main

import "fmt"
import "os"
import "bufio"
import "strings"
import "strconv"

func indexOf(a string, list []string) bool {
    for _, b := range list {
        if b == a {
            return true
        }
    }
    return false
}

func main() {
    scanner := bufio.NewScanner(os.Stdin)

    var N int
    scanner.Scan()
    fmt.Sscan(scanner.Text(),&N)
    
    scanner.Scan()
    TEMPS := scanner.Text()

    var res = strings.Split(TEMPS, " ")
    var val int;
    
    fmt.Fprintln(os.Stderr, res)
    var spaces = strings.Index(TEMPS, " ")
    if(spaces == -1){
        minimum,_ := strconv.Atoi(TEMPS)
        if(val < 0) {
            val = val * -1
        }
        fmt.Println(minimum)
    }else{
        minimum := 9999999
        for i := 0; i < len(res); i++ {
            val, _ = strconv.Atoi(res[i])
            if(val < 0) {
                val = val * -1
            }            
            if(val < minimum){
                minimum = val
            }
        }
        if(indexOf(strconv.Itoa(minimum), res) == false){
            minimum = minimum * -1
        }
            
        fmt.Println(minimum)
    }
}
