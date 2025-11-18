package org.example;

public class Stack {
    int[] array;
    int top, size;
    Object lock;

    public Stack(int n){
        this.array = new int[n];
        this.size = n;
        this.top = -1;
        this.lock = new Object();
    }

    public boolean isEmpty(){
        return top == -1;
    }
    public boolean isFull(){
        return top == (size - 1);
    }
    public  int pop(){
        if (isEmpty()) return  Integer.MAX_VALUE;
        int obj;
        try{ Thread.sleep(1000); } catch (Exception e){ }

        obj = array[top];
        array[top] = Integer.MAX_VALUE;
        top --;

        return  obj;
    }
    public  boolean push(int obj){
        if (isFull()) return  false;

        top ++;

        try{ Thread.sleep(1000); } catch (Exception e) { }

        array[top] = obj;
        return  true;
    }
}
