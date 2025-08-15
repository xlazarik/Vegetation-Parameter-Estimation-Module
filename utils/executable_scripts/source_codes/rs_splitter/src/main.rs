use core::num;
use std::fs::File;
use std::io::{prelude::*, BufReader};
use std::path::Path;
use std::{env, usize};

pub fn file_splitter(parts_dir_path: &str, filename: &str, partitions: i64, mut num_of_lines: i64) {
    println!(">>> OPENING LUT DB");

    let out_path = Path::new(parts_dir_path).join(Path::new("lut_file"));
    println!(">>> COUNTING NO OF LINES IN A FILE");
    if num_of_lines == 0 {
        num_of_lines = get_num_lines(&filename); //  550001 as f64; //
    }
    // let num_of_lines = get_num_lines(&filename) as f64; //  550001 as f64; //
    // let num_lines: f64 = count_lines(File::open(filename).unwrap()).expect("__Error while counting lines__") as f64;
    println!(">>> NUMBER OF LINES {}", num_of_lines);
    let partition_size: i64 = num_of_lines / partitions;
    println!(
        ">>> COMPUTED PARTITION SIZE FOR {} PARTITIONS OF {} LINES IS {}",
        partitions, num_of_lines, partition_size
    );
    let mut intervals: Vec<i64> = vec![0; partitions as usize];
    let mut partition_files: Vec<File> =
        vec![
            File::create(format!("{}_part{}", out_path.to_string_lossy(), 0))
                .expect("__Error while creating partition file__"),
        ];
    println!(">>> CREATING PARTITION FILES");
    for i in 1..partitions {
        intervals[(i as usize)] = intervals[(i - 1) as usize] + partition_size;
        partition_files.push(
            File::create(format!("{}_part{}", out_path.to_string_lossy(), i))
                .expect("__Error while creating partition file__"),
        );
    }
    println!(">>> INTERVALS FOR PARTITIONS ARE: {:?}", intervals);
    let f_handle = File::open(filename).expect("__Error while opening the lut file__");
    let reader_lut = BufReader::new(&f_handle);
    let mut line_count = 0;
    let mut upper_bound = 1;
    println!(">>> WRITING TO PARTITION NO.{}", 0);
    for line in reader_lut.lines() {
        // if line.as_ref().unwrap().starts_with("V") {
        //     continue;
        // }
        if (upper_bound < partitions) && (line_count >= intervals[upper_bound as usize]) {
            // drop(partition_files[upper_bound-1]);
            println!(
                ">>> LINE {} :: WRITING TO PARTITION NO.{}",
                line_count, upper_bound
            );
            upper_bound += 1;
        } else if line_count % 50000 == 0 {
            println!(">>> LINE: {}", line_count);
        }
        writeln!(
            partition_files[(upper_bound - 1) as usize],
            "{}",
            line.expect("__Error while reading line__")
        );
        line_count += 1;
    }
    println!(">>> ALL PARTITIONS SUCCESSFULLY WRITTEN");
    println!("FINISHED");
}

pub fn find_valid_line_interval(
    line: &std::result::Result<String, std::io::Error>,
    low: usize,
    high: usize,
    delim: char,
) -> (usize, usize) {
    let mut total = 0;
    let mut count: usize = 0;
    let mut a = 0;
    let mut b = 0;

    for i in line
        .as_deref()
        .expect("__Error while reading line__")
        .chars()
    {
        if i == delim {
            count += 1;
            if count == low {
                a = total + 1;
            }
            if count == high + 1 {
                b = total;
                break;
            }
        }
        total += 1;
    }
    return (a, b);
}

pub fn file_cropper(
    parts_dir_path: &str,
    filename: &str,
    partitions: i64,
    delim: char,
    has_header: i8,
    low: usize,
    high: usize,
    mut num_of_lines: i64,
) {
    println!(">>> OPENING LUT DB");

    let out_path = Path::new(parts_dir_path).join(Path::new("lut_file"));
    println!(">>> COUNTING NO OF LINES IN A FILE");
    if num_of_lines == 0 {
        num_of_lines = get_num_lines(&filename); //  550001 as f64; //
    }
    // let num_of_lines = get_num_lines(&filename) as f64; //  550001 as f64; //
    // let num_lines: f64 = count_lines(File::open(filename).unwrap()).expect("__Error while counting lines__") as f64;
    println!(">>> NUMBER OF LINES {}", num_of_lines);
    let partition_size: i64 = num_of_lines / partitions;
    println!(
        ">>> COMPUTED PARTITION SIZE FOR {} PARTITIONS OF {} LINES IS {}",
        partitions, num_of_lines, partition_size
    );
    let mut intervals: Vec<i64> = vec![0; partitions as usize];
    let mut partition_files: Vec<File> =
        vec![
            File::create(format!("{}_part{}", out_path.to_string_lossy(), 0))
                .expect("__Error while creating partition file__"),
        ];
    println!(">>> CREATING PARTITION FILES");
    for i in 1..partitions {
        intervals[(i as usize)] = intervals[(i - 1) as usize] + partition_size;
        partition_files.push(
            File::create(format!("{}_part{}", out_path.to_string_lossy(), i))
                .expect("__Error while creating partition file__"),
        );
    }
    println!(">>> INTERVALS FOR PARTITIONS ARE: {:?}", intervals);
    let f_handle = File::open(filename).expect("__Error while opening the lut file__");
    let reader_lut = BufReader::new(&f_handle);
    let mut line_count = 0;
    let mut upper_bound = 1;
    println!(">>> WRITING TO PARTITION NO.{}", 0);
    for line in reader_lut.lines() {
        // if line.as_ref().unwrap().starts_with("V") {
        //     continue;
        // }
        if (upper_bound < partitions) && (line_count >= intervals[upper_bound as usize]) {
            // drop(partition_files[upper_bound-1]);
            println!(
                ">>> LINE {} :: WRITING TO PARTITION NO.{}",
                line_count, upper_bound
            );
            upper_bound += 1;
        } else if line_count % 50000 == 0 {
            println!(">>> LINE: {}", line_count);
        }
        if has_header == 1 && line_count == 0 {
            writeln!(partition_files[(upper_bound - 1) as usize],"{}", line.expect("__Error while reading line__"));
            line_count += 1;
            continue;
        }

        let (a, b) = find_valid_line_interval(&line, low, high, delim);
        writeln!(
            partition_files[(upper_bound - 1) as usize],
            "{}",
            &(line.expect("__Error while reading line__"))[a..b]
        );
        line_count += 1;
    }
    println!(">>> ALL PARTITIONS SUCCESSFULLY WRITTEN");
    println!("FINISHED");
}

fn get_num_lines(filename: &str) -> i64 {
    let mut count = 0;

    let file = File::open(filename).expect("__Error while opening the lut file__");
    let reader_lut = BufReader::new(file);
    for _ in reader_lut.lines() {
        if count % 50000 == 0 {
            println!("NO. OF LINES READ :: {}", count);
        }
        count += 1;
    }
    count
}

fn main() {
    println!("RUNNING");
    let args: Vec<String> = env::args().collect();
    // let mut args = [
    //     "target\\debug\\rs_splitter.exe",
    //     "C:\\dev\\CZGlobe\\mosveg\\X_LUT_DIR\\pseo_prosail_07_03_2021_20_34_45\\lut_partitions",
    //     "C:\\dev\\CZGlobe\\mosveg\\X_LUT_DIR\\pseo_prosail_07_03_2021_20_34_45\\pseo_prosail.csv",
    //     "4",
    //     ",",
    //     "1",
    //     "90",
    //     "600",
    //     "550001",
    // ];
    println!("ARGS: {:?}", args);
    if args.len() < 8 {
        std::process::exit(1);
    }
    let partitions: i64 = args[3]
        .parse()
        .expect("__Error while parsing number of partitions arg__");
    if !Path::new(&args[1]).exists() {
        std::process::exit(1);
    }
    if !Path::new(&args[2]).exists() {
        std::process::exit(2);
    }
    let has_header: i8 = args[5]
        .parse()
        .expect("__Error while parsing header value arg__");
    let low: usize = args[6]
        .parse()
        .expect("__Error while parsing low value arg__");
    let high: usize = args[7]
        .parse()
        .expect("__Error while parsing low value arg__");

    let mut num_of_lines: i64 = 0;
    if args.len() == 9 {
        num_of_lines = args[8]
            .parse()
            .expect("__Error while parsing number of lines arg__");
    }
    println!("LEAVING");
    // file_splitter(&args[1], &args[2], partitions, num_of_lines);
    file_cropper(
        &args[1],
        &args[2],
        partitions,
        args[4].chars().next().unwrap(),
        has_header,
        low,
        high,
        num_of_lines,
    );
}
