import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int[] array = new int[6];
        int max = 0;
        
        for (int i = 0; i < array.length; i++) {
            array[i] = in.nextInt();
            if (array[i] > array[max]) {
                max = i;
            }
        }
        
        int cities = in.nextInt();
        System.out.println(array[max]);
    }
};
