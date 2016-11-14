package org.dlearning;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NonNull;

import java.util.List;

/**
 * Training set
 *
 * @author : Joao Costa (joaocarlosfilho@gmail.com) on 14/11/2016.
 */
@Data
@AllArgsConstructor
public class Training<T> {

    @NonNull
    private List<T> input;
    @NonNull
    private List<T> ouput;
}
