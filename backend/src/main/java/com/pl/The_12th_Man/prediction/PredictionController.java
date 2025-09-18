package com.pl.The_12th_Man.prediction;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/v1")
@CrossOrigin(origins = {"http://localhost:3000", "https://the-12th-man.netlify.app"})
public class PredictionController {

    @Autowired
    private SeasonPredictionRepository predictionRepository;

    @GetMapping("/predictions")
    public ResponseEntity<List<SeasonPrediction>> getPredictions() {
        List<SeasonPrediction> predictions = predictionRepository.findBySeasonOrderByPredictedRankAsc("2025-26");
        return ResponseEntity.ok(predictions);
    }
}