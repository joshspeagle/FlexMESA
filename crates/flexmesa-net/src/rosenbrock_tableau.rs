// SPDX-License-Identifier: LGPL-3.0-or-later
//! Rosenbrock stage tableaux (`Ros3p`, `Ros3pl`, `Rodasp`, `Rodas4`) from
//! Hairer & Wanner's `mod_rosenbrock.f` (`num/private/mod_rosenbrock.f`).

use flexmesa_num::RosenbrockVariant;

pub(crate) struct RodasTableau {
    pub(crate) ns: usize,
    pub(crate) gamma: f64,
    pub(crate) elo: f64,
    pub(crate) no_aux_in_error: bool,
    pub(crate) newf: [bool; 6],
    pub(crate) alpha: [f64; 6],
    pub(crate) ra: [[f64; 6]; 6],
    pub(crate) rc: [[f64; 6]; 6],
    pub(crate) ros_m: [f64; 6],
    pub(crate) ros_e: [f64; 6],
}

pub(crate) fn tableau_for(variant: RosenbrockVariant) -> RodasTableau {
    let mut tab = RodasTableau {
        ns: 6,
        gamma: 0.25,
        elo: 4.0,
        no_aux_in_error: false,
        newf: [true; 6],
        alpha: [0.0, 0.386, 0.210, 0.630, 1.0, 1.0],
        ra: [[0.0; 6]; 6],
        rc: [[0.0; 6]; 6],
        ros_m: [0.0; 6],
        ros_e: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    };
    match variant {
        RosenbrockVariant::Ros3p => {
            tab.ns = 3;
            tab.gamma = 0.78867513459481286;
            tab.elo = 3.0;
            tab.no_aux_in_error = true;
            tab.newf = [true, true, false, false, false, false];
            tab.alpha = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
            tab.ra[1][0] = 1.26794919243112273;
            tab.ra[2][0] = 1.26794919243112273;
            tab.rc[1][0] = -1.60769515458673630;
            tab.rc[2][0] = -3.46410161513775469;
            tab.rc[2][1] = -1.73205080756887734;
            tab.ros_m[0..3].copy_from_slice(&[
                2.0,
                0.57735026918962578,
                0.42264973081037424,
            ]);
            tab.ros_e = [
                2.0 - 2.11324865405187124,
                0.57735026918962578 - 1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ];
        }
        RosenbrockVariant::Ros3pl => {
            tab.ns = 4;
            tab.gamma = 0.435866521508459;
            tab.elo = 3.0;
            tab.no_aux_in_error = true;
            tab.newf = [true, true, true, false, false, false];
            tab.alpha = [0.0, 0.5, 1.0, 1.0, 0.0, 0.0];
            tab.ra[1][0] = 1.147140180139521;
            tab.ra[2][0] = 2.463070773030053;
            tab.ra[2][1] = 1.147140180139521;
            tab.ra[3][0] = 2.463070773030053;
            tab.ra[3][1] = 1.147140180139521;
            tab.rc[1][0] = -2.631861185781065;
            tab.rc[2][0] = -1.302364158113095;
            tab.rc[2][1] = 2.769432022251304;
            tab.rc[3][0] = -1.552568958732400;
            tab.rc[3][1] = 2.587743501215153;
            tab.rc[3][2] = -1.416993298352020;
            tab.ros_m[0..4].copy_from_slice(&[
                2.463070773030053,
                1.147140180139521,
                0.0,
                1.0,
            ]);
            tab.ros_e = [
                2.463070773030053 - 2.346947683513665,
                1.147140180139521 - 0.456530569451895,
                -0.056949243945495,
                1.0 - 0.738684936166224,
                0.0,
                0.0,
            ];
        }
        RosenbrockVariant::Rodasp => {
            tab.alpha = [0.0, 0.750, 0.210, 0.630, 1.0, 1.0];
            tab.ra[1][0] = 3.0;
            tab.ra[2][0] = 0.1831036793486759e1;
            tab.ra[2][1] = 0.4955183967433795;
            tab.ra[3][0] = 0.2304376582692669e1;
            tab.ra[3][1] = -0.5249275245743001e-1;
            tab.ra[3][2] = -0.1176798761832782e1;
            tab.ra[4][0] = -0.7170454962423024e1;
            tab.ra[4][1] = -0.4741636671481785e1;
            tab.ra[4][2] = -0.1631002631330971e2;
            tab.ra[4][3] = -0.1062004044111401e1;
            let ra4 = [tab.ra[4][0], tab.ra[4][1], tab.ra[4][2], tab.ra[4][3]];
            tab.ra[5][0..4].copy_from_slice(&ra4);
            tab.ra[5][4] = 1.0;

            tab.rc[1][0] = -12.0;
            tab.rc[2][0] = -0.8791795173947035e1;
            tab.rc[2][1] = -0.2207865586973518e1;
            tab.rc[3][0] = 0.1081793056857153e2;
            tab.rc[3][1] = 0.6780270611428266e1;
            tab.rc[3][2] = 0.1953485944642410e2;
            tab.rc[4][0] = 0.3419095006749676e2;
            tab.rc[4][1] = 0.1549671153725963e2;
            tab.rc[4][2] = 0.5474760875964130e2;
            tab.rc[4][3] = 0.1416005392148534e2;
            tab.rc[5][0] = 0.3462605830930532e2;
            tab.rc[5][1] = 0.1530084976114473e2;
            tab.rc[5][2] = 0.5699955578662667e2;
            tab.rc[5][3] = 0.1840807009793095e2;
            tab.rc[5][4] = -0.5714285714285717e1;

            tab.ros_m[0..4].copy_from_slice(&tab.ra[4][0..4]);
            tab.ros_m[4] = 1.0;
            tab.ros_m[5] = 1.0;
        }
        RosenbrockVariant::Rodas4
        | RosenbrockVariant::Rodas3
        | RosenbrockVariant::Ros2
        | RosenbrockVariant::Rose2 => {
            tab.ra[1][0] = 1.544;
            tab.ra[2][0] = 0.9466785280815826;
            tab.ra[2][1] = 0.2557011698983284;
            tab.ra[3][0] = 0.3314825187068521e1;
            tab.ra[3][1] = 0.2896124015972201e1;
            tab.ra[3][2] = 0.9986419139977817;
            tab.ra[4][0] = 0.1221224509226641e1;
            tab.ra[4][1] = 0.6019134481288629e1;
            tab.ra[4][2] = 0.1253708332932087e2;
            tab.ra[4][3] = -0.6878860361058950;
            let ra4 = [tab.ra[4][0], tab.ra[4][1], tab.ra[4][2], tab.ra[4][3]];
            tab.ra[5][0..4].copy_from_slice(&ra4);
            tab.ra[5][4] = 1.0;

            tab.rc[1][0] = -5.6688;
            tab.rc[2][0] = -0.2430093356833875e1;
            tab.rc[2][1] = -0.2063599157091915;
            tab.rc[3][0] = -0.1073529058151375;
            tab.rc[3][1] = -0.9594562251023355e1;
            tab.rc[3][2] = -0.2047028614809616e2;
            tab.rc[4][0] = 0.7496443313967647e1;
            tab.rc[4][1] = -0.1024680431464352e2;
            tab.rc[4][2] = -0.3399990352819905e2;
            tab.rc[4][3] = 0.1170890893206160e2;
            tab.rc[5][0] = 0.8083246795921522e1;
            tab.rc[5][1] = -0.7981132988064893e1;
            tab.rc[5][2] = -0.3152159432874371e2;
            tab.rc[5][3] = 0.1631930543123136e2;
            tab.rc[5][4] = -0.6058818238834054e1;

            tab.ros_m[0..4].copy_from_slice(&tab.ra[4][0..4]);
            tab.ros_m[4] = 1.0;
            tab.ros_m[5] = 1.0;
        }
    }
    tab
}
