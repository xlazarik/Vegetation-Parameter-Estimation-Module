use std::env;
use std::fs::File;
use serde_json::{Result, Value};
use std::io::Write;
use std::path::Path;

type Record = Vec<f64>;

struct Band {
    name: String,
    indices: Vec<i64>,
    coefficients: Vec<f64>,
    sum: f64,
}

fn convolve(
    srf: Vec<Band>,
    lut_path: &String,
    new_path: &String,
) {
    let mut resampled_file =
        File::create(new_path).expect("Error while creating file resampledSatLut");

    let lut_db = File::open(lut_path).expect("__Error while opening the lut file__");
    let mut lut_reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(lut_db);
    let mut id: i64 = 0;
    let mut bands_value: f64 = 0.0;


    println!("RESAMPLING");
    writeln!(resampled_file, "443.0,490.0,560.0,665.0,705.0,740.0,783.0,842.0,865.0,945.0,1375.0,1610.0,2190.0");

    for i in lut_reader.deserialize() {
        let record: Record = i.expect("Error while parsing lut_reader record");

        for band_i in 0..srf.len() {
            bands_value = 0.0;
            for index in &srf[band_i].indices {
                bands_value += record[*index as usize] * srf[band_i].coefficients[(*index - srf[band_i].indices[0]) as usize];
            }
            write!(resampled_file, "{:.6}", (bands_value / srf[band_i].sum));
            write!(resampled_file, "{}", if band_i == srf.len() - 1 { '\n' } else { ',' });
        }
        id += 1;
        if id % 25000 == 0 {
            println!("{}", id);
        }
    }
}


fn main() {
    println!("RUNNING");
    let args: Vec<String> = env::args().collect();
    println!("ARGS: {:?}", args);
    if args.len() != 4 {
        std::process::exit(1);
    }

    if !Path::new(&args[1]).exists() {
        std::process::exit(1);
    }

    let s = &args[3];
    // let s = &*s.replace("\\","");

    let v: Value = serde_json::from_str(&s[0..s.len()].replace("\\", "")).unwrap();
    let mut srf = Vec::new();

    for (name, obj) in v.as_object().unwrap().iter() {
        let mut band: Band = Band { name: name.clone(), indices: Vec::new(), coefficients: Vec::new(), sum: 0.0 };
        let ind = obj.as_array().unwrap()[0].as_array().unwrap();
        let coeffs = obj.as_array().unwrap()[1].as_array().unwrap();
        for i in 0..ind.len() {
            band.indices.push(ind[i].as_i64().unwrap());
            band.coefficients.push(coeffs[i].as_f64().unwrap());
        }
        band.sum = band.coefficients.iter().sum();
        srf.push(band);
    }
    convolve(srf, &args[1], &args[2]);

    println!("FINISHED");
}
