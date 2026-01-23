package com.example;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class ResourceHandler {

    public void processFile(String filePath) throws IOException {
        File file = new File(filePath);
        //Using try-with-resources to ensure resources are closed
        try (InputStream inputStream = new FileInputStream(file);
             OutputStream outputStream = new FileOutputStream(file.getAbsolutePath() + ".processed")) {

            byte[] buffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }

        } catch (IOException e) {
            // Log the exception, potentially re-throw, or handle appropriately
            System.err.println("Error processing file: " + e.getMessage());
            throw e; // Re-throwing to propagate the exception if necessary
        }
        //No need to close streams explicitly; try-with-resources handles it.
    }

    public void allocateMemory(int size) {
        byte[] memory = new byte[size];
        // Potentially use memory here

        // Memory is automatically reclaimed by garbage collection when the method ends
        // However, for large allocations in long-running processes, consider setting the
        // reference to null to hint to the garbage collector.  This isn't always necessary.
    }

    public void closeExternalResource(AutoCloseable resource) throws Exception {
        if (resource != null) {
            try {
                resource.close();
            } catch (Exception e) {
                System.err.println("Error closing resource: " + e.getMessage());
                throw e; // Re-throwing to propagate the exception if necessary
            }
        }
    }
}
