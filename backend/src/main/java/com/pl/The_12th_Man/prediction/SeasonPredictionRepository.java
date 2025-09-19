package com.pl.The_12th_Man.prediction;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface SeasonPredictionRepository extends JpaRepository<SeasonPrediction, Long> {
    List<SeasonPrediction> findBySeasonOrderByPredictedRankAsc(String season);
}