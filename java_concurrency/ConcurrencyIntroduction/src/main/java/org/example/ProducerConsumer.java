package org.example;
import java.util.Queue;
import java.util.LinkedList;

public class ProducerConsumer {
    private Queue<Integer> q;
    int capacity;

    public ProducerConsumer(int capacity){
        q = new LinkedList<>();
        this.capacity = capacity;
    }

    public boolean add(int elem)
    {
        synchronized (q)
        {
            while (q.size() == capacity) {
                try {
                    wait();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            q.add(elem);
            notifyAll();
            return true;
        }
    }
    public int kick()
    {
        synchronized (q)
        {
            while (q.size() == 0) {
                try {
                    wait();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            int head = q.poll(); //q.remove() not throw exception if q is empty
            notifyAll();
            return head;
        }
    }

    }

