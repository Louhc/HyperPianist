//! Distributed Dory PCS Benchmark
//!
//! Tests only the PCS (commit + open + verify), not the full SNARK.

use arithmetic::math::Math;
use ark_bn254::{Bn254, Fr};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::CanonicalSerialize;
use ark_std::UniformRand;
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};
use std::{sync::Arc, time::{Duration, Instant}};
use subroutines::pcs::{
    prelude::{DeDory, DeDorySRS, PCSError},
    PolynomialCommitmentScheme,
};

mod common;
use common::{barrier, d_evaluate_mle, test_rng, test_rng_deterministic, Opt};

fn main() {
    common::network_run(|opt: Opt| {
        bench_dory_pcs(opt.mu, opt.iterations).unwrap();
    });
}

fn bench_dory_pcs(
    mu: usize,
    iterations: usize,
) -> Result<(), PCSError> {
    let mut rng = test_rng();
    let num_party = Net::n_parties();
    let num_party_vars = num_party.log_2();
    let nv = mu - num_party_vars; // local variables per party
    let is_master = Net::am_master();

    macro_rules! master_print {
        ($($arg:tt)*) => { if is_master { println!($($arg)*); } };
    }

    master_print!("========================================");
    master_print!("dDory PCS Distributed Benchmark");
    master_print!("  mu = {}, local_nv = {}, parties = {}, iterations = {}", mu, nv, num_party, iterations);
    master_print!("========================================");

    // Generate or load SRS
    let start = Instant::now();
    let srs = {
        let mut srs_rng = test_rng_deterministic();
        DeDory::<Bn254>::gen_srs_for_testing(&mut srs_rng, mu)?
    };
    master_print!("Gen SRS: {:?}", start.elapsed());

    // Trim parameters
    let start = Instant::now();
    let (ck, vk) = DeDory::trim(&srs, None, Some(mu))?;
    master_print!("Trim: {:?}", start.elapsed());

    // Run iterations
    let mut commit_times = Vec::with_capacity(iterations);
    let mut open_times = Vec::with_capacity(iterations);
    let mut verify_times = Vec::with_capacity(iterations);
    let mut proof_size = 0usize;

    // Reset stats before iterations to capture all communication
    Net::reset_stats();

    for iter in 0..iterations {
        master_print!("\n--- Iteration {} ---", iter + 1);

        // Generate random polynomial (each party generates its own share)
        let poly = Arc::new(DenseMultilinearExtension::rand(nv, &mut rng));

        // Generate random evaluation point (master broadcasts)
        let point: Vec<Fr> = if is_master {
            let point: Vec<Fr> = (0..mu).map(|_| Fr::rand(&mut rng)).collect();
            Net::recv_from_master_uniform(Some(point))
        } else {
            Net::recv_from_master_uniform(None)
        };

        // Commit
        barrier();
        let start = Instant::now();
        let (com, advice) = DeDory::d_commit(&ck, &poly)?;
        let commit_time = start.elapsed();
        commit_times.push(commit_time);

        // Open
        barrier();
        let start = Instant::now();
        let proof = DeDory::open(&ck, &poly, &advice, &point)?;
        let open_time = start.elapsed();
        open_times.push(open_time);

        // Verify (master only)
        if is_master {
            let com = com.unwrap();
            let value = d_evaluate_mle(&poly, Some(&point)).unwrap();

            let start = Instant::now();
            let result = DeDory::verify(&vk, &com, &point, &value, &proof)?;
            let verify_time = start.elapsed();
            verify_times.push(verify_time);
            assert!(result, "Verification failed at iteration {}", iter + 1);

            // Record proof size (last iteration)
            if iter == iterations - 1 {
                let mut proof_bytes = Vec::new();
                proof.serialize_compressed(&mut proof_bytes).unwrap();
                proof_size = proof_bytes.len();
            }

            master_print!("Commit: {:?}, Open: {:?}, Verify: {:?}",
                commit_time, open_time, verify_time);

            // Machine-readable per-iteration output
            println!("ITER_{}_COMMIT_MS: {:.3}", iter + 1, commit_time.as_secs_f64() * 1000.0);
            println!("ITER_{}_OPEN_MS: {:.3}", iter + 1, open_time.as_secs_f64() * 1000.0);
            println!("ITER_{}_VERIFY_MS: {:.3}", iter + 1, verify_time.as_secs_f64() * 1000.0);
        } else {
            // Workers still need to participate in d_evaluate_mle
            d_evaluate_mle(&poly, None);
        }
    }

    // Get average communication stats per iteration (master's sent + received)
    let stats = Net::stats();
    let total_comm_bytes = (stats.bytes_sent + stats.bytes_recv) as f64;

    // Print summary (master only)
    if is_master {
        let avg = |times: &[Duration]| -> Duration {
            times.iter().sum::<Duration>() / times.len() as u32
        };

        let avg_comm_mb = total_comm_bytes / iterations as f64 / (1024.0 * 1024.0);

        master_print!("\n========================================");
        master_print!("Summary ({} iterations):", iterations);
        master_print!("  Commit (avg): {:?}", avg(&commit_times));
        master_print!("  Open (avg):   {:?}", avg(&open_times));
        master_print!("  Verify (avg): {:?}", avg(&verify_times));
        master_print!("  Proof size:   {:.2} KB", proof_size as f64 / 1024.0);
        master_print!("  Communication: {:.2} MB", avg_comm_mb);
        master_print!("========================================");

        // Machine-readable output
        println!("COMMIT_TIME_MS: {:.3}", avg(&commit_times).as_secs_f64() * 1000.0);
        println!("OPEN_TIME_MS: {:.3}", avg(&open_times).as_secs_f64() * 1000.0);
        println!("VERIFY_TIME_MS: {:.3}", avg(&verify_times).as_secs_f64() * 1000.0);
        println!("PROOF_SIZE_KB: {:.2}", proof_size as f64 / 1024.0);
        println!("COMM_TOTAL_BYTES: {}", (total_comm_bytes / iterations as f64) as u64);

        // Combined prover time
        let prover_ms = avg(&commit_times).as_secs_f64() * 1000.0
            + avg(&open_times).as_secs_f64() * 1000.0;
        println!("PROVER_TIME_MS: {:.3}", prover_ms);
    }

    Ok(())
}
