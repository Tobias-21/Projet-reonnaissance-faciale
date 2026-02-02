package com.example.demo.Service;

import com.example.demo.Model.FaceEmbedding;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.http.client.ReactorClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;
import reactor.netty.http.client.HttpClient;

import java.io.IOException;
import java.time.Duration;
import java.util.List;

@Component
public class PythonClient {

    //private RestTemplate restTemplate = new RestTemplate();
    private final RestClient restClient;
    public PythonClient() {
        HttpClient httpClient = HttpClient.create()
                .responseTimeout(Duration.ofMinutes(2));
        this.restClient = RestClient.builder().baseUrl("http://localhost:8000").requestFactory(new ReactorClientHttpRequestFactory(httpClient)).build();
    }

    public List<FaceResponse> getSignature(List<MultipartFile> files) throws IOException {
        MultipartBodyBuilder builder = new MultipartBodyBuilder();

        for (MultipartFile file : files) {
            MediaType mediaType = (file.getContentType() != null)
                    ? MediaType.parseMediaType(file.getContentType())
                    : MediaType.APPLICATION_OCTET_STREAM;

            builder.part("files", new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            }).contentType(mediaType);
        }


        return restClient.post()
                .uri("/detection") // Le nom de votre route FastAPI
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(builder.build())
                .retrieve()
                .body(new ParameterizedTypeReference<List<FaceResponse>>() {});

        //return (response != null) ? response.signature() : null;
    }

}
